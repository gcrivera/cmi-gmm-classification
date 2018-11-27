addpath(genpath('matlab-reqs'))

nmix = 512;
model_path = 'models/512c_13f/';

% Load training data
disp('Loading data...')
cmi1_data = readNPY('data/train_cmi1_13f.npy');
cmi2_data = readNPY('data/train_cmi2_13f.npy');
cmi3_data = readNPY('data/train_cmi3_13f.npy');
cmi4_data = readNPY('data/train_cmi4_13f.npy');
cmi5_data = readNPY('data/train_cmi5_13f.npy');

all_data = cat(1, cmi1_data, cmi2_data, cmi3_data, cmi4_data, cmi5_data);

% Generate UBM
disp('Generating UBM...');
ubm = gmm_em(all_data, nmix, 10, 1, 4, strcat(model_path, 'ubm.mat'));

% Generate GMMs

disp('Adapting CMI1...')
cmi1_gmm = mapAdapt(cmi1_data, ubm, 10, 'm', strcat(model_path, 'cmi1.mat'));
disp('Adapting CMI2...')
cmi2_gmm = mapAdapt(cmi2_data, ubm, 10, 'm', strcat(model_path, 'cmi2.mat'));
disp('Adapting CMI3...')
cmi3_gmm = mapAdapt(cmi3_data, ubm, 10, 'm', strcat(model_path, 'cmi3.mat'));
disp('Adapting CMI4...')
cmi4_gmm = mapAdapt(cmi4_data, ubm, 10, 'm', strcat(model_path, 'cmi4.mat'));
disp('Adapting CMI5...')
cmi5_gmm = mapAdapt(cmi5_data, ubm, 10, 'm', strcat(model_path, 'cmi5.mat'));