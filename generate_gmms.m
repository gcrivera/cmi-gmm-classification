addpath(genpath('matlab-reqs'))

nmix = 512;
model_path = 'models/mfcc_phoneme/512c/';

% Load training data
disp('Loading data...')
cmi1_data = double(readNPY('data/mfcc_phoneme/train_cmi1_20f.npy')');
cmi2_data = double(readNPY('data/mfcc_phoneme/train_cmi2_20f.npy')');
cmi3_data = double(readNPY('data/mfcc_phoneme/train_cmi3_20f.npy')');
cmi4_data = double(readNPY('data/mfcc_phoneme/train_cmi4_20f.npy')');
cmi5_data = double(readNPY('data/mfcc_phoneme/train_cmi5_20f.npy')');

% all_data = num2cell(cat(1, cmi1_data, cmi2_data, cmi3_data, cmi4_data, cmi5_data)', 1);
all_data = {cmi1_data; cmi2_data; cmi3_data; cmi4_data; cmi5_data};
% Generate UBM
disp('Generating UBM...');
ubm = gmm_em(all_data, nmix, 10, 1, 4, strcat(model_path, 'ubm.mat'));
% Generate GMMs

disp('Adapting CMI1...')
cmi1_gmm = mapAdapt(num2cell(cmi1_data, 1), ubm, 10, 'm', strcat(model_path, 'cmi1.mat'));
disp('Adapting CMI2...')
cmi2_gmm = mapAdapt(num2cell(cmi2_data, 1), ubm, 10, 'm', strcat(model_path, 'cmi2.mat'));
disp('Adapting CMI3...')
cmi3_gmm = mapAdapt(num2cell(cmi3_data, 1), ubm, 10, 'm', strcat(model_path, 'cmi3.mat'));
disp('Adapting CMI4...')
cmi4_gmm = mapAdapt(num2cell(cmi4_data, 1), ubm, 10, 'm', strcat(model_path, 'cmi4.mat'));
disp('Adapting CMI5...')
cmi5_gmm = mapAdapt(num2cell(cmi5_data, 1), ubm, 10, 'm', strcat(model_path, 'cmi5.mat'));

exit