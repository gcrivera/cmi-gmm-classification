addpath(genpath('matlab-reqs'))

model_path = 'models/512c_13f/';

% Load test data
cmi1_data = readNPY('data/test_cmi1_13f.npy');
cmi2_data = readNPY('data/test_cmi2_13f.npy');
cmi3_data = readNPY('data/test_cmi3_13f.npy');
cmi4_data = readNPY('data/test_cmi4_13f.npy');
cmi5_data = readNPY('data/test_cmi5_13f.npy');

% Set up args
models = {load(strcat(model_path, 'cmi1.mat')), load(strcat(model_path, 'cmi2.mat')), load(strcat(model_path, 'cmi3.mat')), load(strcat(model_path, 'cmi4.mat')), load(strcat(model_path, 'cmi5.mat'))};
tests = {cmi1_data, cmi2_data, cmi3_data, cmi4_data, cmi5_data};
trials = [  1 1; 1 2; 1 3; 1 4; 1 5;
            2 1; 2 2; 2 3; 2 4; 2 5;
            3 1; 3 2; 3 3; 3 4; 3 5;
            4 1; 4 2; 4 3; 4 4; 4 5;
            5 1; 5 2; 5 3; 5 4; 5 5  ];

% Generate scores

scores = score_gmm_trials(models, tests, trials, strcat(model_path, 'ubm.mat'));

% View confusion matrix
% Not sure what scores outputs
% title('CMI Likelihood (GMM Model)');
% ylabel('Test # (True CMI)'); xlabel('Model #');
% colorbar; drawnow; axis xy
% figure
% eer = compute_eer(scores, answers, false);