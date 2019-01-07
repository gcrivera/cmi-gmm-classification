addpath(genpath('matlab-reqs'))

model_path = 'models/t4_512c_20f/';

% Load test data
disp('Loading data...')
cmi1_data = readNPY('data/t4_test_cmi1_20f.npy');
cmi2_data = readNPY('data/t4_test_cmi2_20f.npy');
cmi3_data = readNPY('data/t4_test_cmi3_20f.npy');
cmi4_data = readNPY('data/t4_test_cmi4_20f.npy');
cmi5_data = readNPY('data/t4_test_cmi5_20f.npy');
all_data = { cmi1_data; cmi2_data; cmi3_data; cmi4_data; cmi5_data };

ubm = load(strcat(model_path, 'ubm.mat'));
cmi1 = load(strcat(model_path, 'cmi1.mat'));
cmi2 = load(strcat(model_path, 'cmi2.mat'));
cmi3 = load(strcat(model_path, 'cmi3.mat'));
cmi4 = load(strcat(model_path, 'cmi4.mat'));
cmi5 = load(strcat(model_path, 'cmi5.mat'));
models = {cmi1.gmm, cmi2.gmm, cmi3.gmm, cmi4.gmm, cmi5.gmm};

disp('Calculating Results')

true_labels = [];
predicted_labels = [];
matching = 0;

for i = 1:5
    data = cell2mat(all_data(i));
    data_size = size(data);
    tests = {};
    for j = 1:data_size(1)
        utterance_data = squeeze(data(j,:,:));
        utterance_data = utterance_data(any(utterance_data,2),:);
        tests{end+1} = utterance_data';
    end
    
    trials = zeros(5*data_size(1), 2);
    for j = 1:data_size(1)
        trials((j-1)*5+1:j*5,1) = 1:5;
        trials((j-1)*5+1:j*5,2) = j;
    end
    
    % should be same size as num rows of trials
    % scores((utterance_num-1)*5 + model #) gives utterance score for model
    scores = score_gmm_trials(models, tests, trials, ubm.gmm);
    
    for j = 1:data_size(1)
        partial_idx = (j-1)*5;
        utterance_scores = scores(1+partial_idx:5+partial_idx, 1);
        [val, idx] = max(utterance_scores);
        true_labels(end+1) = i;
        predicted_labels(end+1) = idx;
        if i == idx
            matching = matching + 1;
        end
    end
end

labels = size(true_labels);
disp('Accuracy:')
disp(double(matching) / double(labels(2)))
cm = confusionchart(true_labels, predicted_labels, 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');