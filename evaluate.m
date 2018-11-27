addpath(genpath('matlab-reqs'))

model_path = 'models/512c_13f/';

% Load test data
cmi1_data = readNPY('data/test_cmi1_13f.npy');
cmi2_data = readNPY('data/test_cmi2_13f.npy');
cmi3_data = readNPY('data/test_cmi3_13f.npy');
cmi4_data = readNPY('data/test_cmi4_13f.npy');
cmi5_data = readNPY('data/test_cmi5_13f.npy');
all_data = [ cmi1_data cmi2_data cmi3_data cmi4_data cmi5_data ];

% Set up args
models = {load(strcat(model_path, 'cmi1.mat')), load(strcat(model_path, 'cmi2.mat')), load(strcat(model_path, 'cmi3.mat')), load(strcat(model_path, 'cmi4.mat')), load(strcat(model_path, 'cmi5.mat'))};

for i = 1:5
    data = all_data(i);
    data_size = size(data);
    tests = {};
    for j = 1:data_size(1)
        utterance_data = data(j);
        utterance_data = utterance_data(:,any(data,1));
        tests{end+1} = utterance_data;
    end
    
    trials = zeros(5*data_size(1), 2);
    for j = 1:data_size(1)
        trials((j-1)*5+1:j*5,1) = 1:5;
        trials((j-1)*5+1:j*5,2) = j;
    end
    
    % should be same size as num rows of trials
    % scores((utterance_num-1)*5 + model #) gives utterance score for model
    scores = score_gmm_trials(models, tests, trials, strcat(model_path, 'ubm.mat'));
    
end


% View confusion matrix
% Not sure what scores outputs
% title('CMI Likelihood (GMM Model)');
% ylabel('Test # (True CMI)'); xlabel('Model #');
% colorbar; drawnow; axis xy
% figure
% eer = compute_eer(scores, answers, false);