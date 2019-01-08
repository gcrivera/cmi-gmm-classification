import cmi
import librosa
import numpy as np
import soundfile as sf
from random import shuffle
from tqdm import tqdm

def extract(num_features, phoneme_feat=False):
    file_locations = get_file_locations()
    if phoneme_feat:
        phoneme_data = get_phonemes()
    max_length = 8265

    transcription_file = open('data/text.bw')
    transcription_lines = transcription_file.readlines()
    transcription_file.close()

    shuffle(transcription_lines)

    max,min = calculate_cmi_norm(transcription_lines)

    test_idx = {'1': 7900, '2': 4556, '3': 4840, '4': 2552, '5': 1707}

    train_cmi = {'1': [], '2': [], '3': [], '4': [], '5': []}
    test_cmi = {'1': [], '2': [], '3': [], '4': [], '5': []}

    print('Generating features...')
    for line in tqdm(transcription_lines):

        cmi_class = cmi.calculate(line, norm=(max,min))
        if cmi_class == None:
            continue

        line_data = line.split()
        utterance_data = line_data[0]
        utterance_words = line_data[1:]

        utterance_data_list = utterance_data.split('_')
        file = '_'.join(utterance_data_list[:-2])
        start = float(utterance_data_list[-2])
        stop = float(utterance_data_list[-1])
        file_location = file_locations[file]

        if phoneme_feat:
            phonemes = phoneme_data[utterance_data]
            # TODO: need to change this fn
            y_phoneme = get_phoneme_feature(phonemes)

        y, sr = sf.read(file_location, start=int(16000*start), stop=int(16000*stop)+1)
        # each column represents 0.01 second step
        # mfcc = librosa.feature.mfcc(y, sr, n_mfcc=num_features, n_fft=400, hop_length=160, fmin=133, fmax=6955)
        Y = np.abs(librosa.core.stft(y, n_fft=400, hop_length=160))
        # spec_delta = librosa.feature.delta(spec)
        # spec_delta_delta = librosa.feature.delta(spec, order=2)
        # Y = np.concatenate((spec, spec_delta)) #, spec_delta_delta))

        # TODO: check if len(phonemes) and Y.shape are the same
        # if so, then create one hot of phoneme features and concatenate,
        # otherwise gonna have to step through time stamps and match them up with
        # the 25 ms sliding window
        print('phoneme shape')
        print(y_phoneme.shape)
        print('Y shape')
        print(Y.shape)
        exit()
        Y = cmvn_slide(Y, cmvn='m').T

        if len(train_cmi[cmi_class]) < test_idx[cmi_class]:
            train_cmi[cmi_class].append(Y)
        else:
            pad_utterance = np.zeros((max_length - Y.shape[0], Y.shape[1])) # num_features*3, Take out *3 if not using deltas
            test_cmi[cmi_class].append(np.concatenate((Y, pad_utterance)))

    for i in range(5):
        cmi_class = str(i+1)
        np.save('data/spec/train_cmi' + cmi_class + '_' + str(num_features) + 'f.npy', np.concatenate(train_cmi[cmi_class]))
        np.save('data/spec/test_cmi' + cmi_class + '_' + str(num_features) + 'f.npy', test_cmi[cmi_class])

def get_file_locations():
    audio_locations = open('data/wav_train.scp')
    audio_location_lines = audio_locations.readlines()
    audio_locations.close()

    locations = {}
    print('Loading file locations...')
    for line in tqdm(audio_location_lines):
        line_data = line.split()
        if len(line_data) > 2:
            locations[line_data[0]] = line_data[2]

    return locations

def get_phonemes():
    phonemes = open('data/phoneme_HU.mlf')
    phoneme_lines = phonemes.readlines()
    phonemes.close()

    phoneme_dict = {}
    new_recording = True
    rec_name = ''
    print('Loading phoneme data...')
    for line in tqdm(phoneme_lines):
        if new_recording:
            rec_name = line[:-7]
            phoneme_dict[rec_name] = []
            new_recording = False
        elif line.startswith('.'):
            new_recording = True
        else:
            phoneme_data = line.split()
            phoneme_dict[rec_name].append((phoneme_data[2], (phoneme_data[0][:-5], phoneme_data[1][:-5])))

    return phoneme_dict

def get_phoneme_feature(phonemes):
    phoneme_feature_locations = {'pau': 0, 'v': 1, 'O': 2, 'n': 3, 'e:': 4, 'k': 5,
     'E': 6, 'int': 7, 's': 8, 'o': 9, 'm': 10, 't': 11, 'l': 12, 'd': 13, 'u': 14,
     'A:': 15, 'J': 16, 'h': 17,'o:': 18, 'i': 19, 'S': 20, 'z': 21, 'j': 22, 'i:':23,
     'd_': 24, 'r': 25, 'g': 26, '_2': 27, 'b': 28, 'u:': 29, 'y': 30, 'spk': 31,
     'p': 32, 'h1': 33, 'n:': 34, 'f': 35, ':2': 36, 'tS': 37, 'N': 38, 'm:': 39,
     'y:': 40, 'ts_': 41, 'l:': 42, 'b:': 43, 's:': 44, 'ts': 45, 'Z': 46, 't:': 47,
     'j:': 48, 'd_:': 49, 'z:': 50, 't1': 51, 't1:': 52, 'r:': 53, 'tS_': 54, 'J:': 55,
     'x': 56, 'k:': 57, 'dz': 58, 'F': 59, 'S:': 60}

    features = []
    start = 0
    end = phonemes[-1][1][1]
    while start < end:
        feature = np.zeros(60)
        for phoneme in phonemes:
            phone_start = phoneme[1][0]
            phone_end = phoneme[1][1]
            if (phone_start >= start and phone_start <= start+25)
                or (phone_end >= start and phone_end <= start+25)
                or (phone_start <= start and phone_end >= start+25):
                feature[phoneme_feature_locations[phoneme[0]]] = 1
        features.append(feature)
        start += 10

    return np.appary(features)

def calculate_cmi_norm(transcription_lines):
    cmis = []
    for line in transcription_lines:
        cmi_val = cmi.calculate(line)
        cmis.append(cmi_val)

    return (float(np.amax(cmis)), float(np.amin(cmis)))

def cmvn_slide(X, win_len=300, cmvn=False):
    max_length = np.shape(X)[0]
    new_feat = np.empty_like(X)
    cur = 1
    left_win = 0
    right_win = int(win_len/2)

    for cur in range(max_length):
        cur_slide = X[cur-left_win:cur+right_win,:]
        mean = np.mean(cur_slide,axis=0)
        std = np.std(cur_slide,axis=0)
        if cmvn == 'mv':
            new_feat[cur,:] = (X[cur,:]-mean)/std # for cmvn
        elif cmvn == 'm':
            new_feat[cur,:] = (X[cur,:]-mean) # for cmn
        if left_win < win_len/2:
            left_win += 1
        elif max_length-cur < win_len/2:
            right_win -= 1
    return new_feat