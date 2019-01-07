import cmi
import librosa
import numpy as np
import soundfile as sf
from random import shuffle
from tqdm import tqdm

def extract(num_features):
    file_locations = get_file_locations()
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

        y, sr = sf.read(file_location, start=int(16000*start), stop=int(16000*stop)+1)
        # each column represents 0.01 second step
        # mfcc = librosa.feature.mfcc(y, sr, n_mfcc=num_features, n_fft=400, hop_length=160, fmin=133, fmax=6955)
        spec = np.abs(librosa.core.stft(y, n_fft=400, hop_length=160))
        spec_delta = librosa.feature.delta(spec)
        spec_delta_delta = librosa.feature.delta(spec, order=2)
        Y = np.concatenate((spec, spec_delta, spec_delta_delta))
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