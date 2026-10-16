% CLASSIFY_NN_KNN  Nearest Neighbour and K-Nearest Neighbour classification
% of 10 Caltech-101 categories using HOG features.
%
% Run from the src/ folder. Uses the feature files in ../features.
% Requires MATLAB R2016b or later (implicit expansion). No toolboxes needed.

clc; clear;

featureDir = fullfile('..', 'features');
K = 5;   % neighbours used by KNN

classes = {'sunflower', 'crab', 'chair', 'cellphone', 'dolphin', ...
           'stopsign', 'helicopter', 'lamp', 'kangaroo', 'laptop'};
nClasses = numel(classes);

%% Load training and test features
Xtrain = []; ytrain = [];
Xtest  = []; ytest  = [];
for c = 1:nClasses
    tr = load_features(fullfile(featureDir, [classes{c} '_featurevector.mat']));
    te = load_features(fullfile(featureDir, [classes{c} 'Test_featurevector.mat']));
    Xtrain = [Xtrain; tr];                    %#ok<AGROW>
    ytrain = [ytrain; c * ones(size(tr, 1), 1)]; %#ok<AGROW>
    Xtest  = [Xtest;  te];                    %#ok<AGROW>
    ytest  = [ytest;  c * ones(size(te, 1), 1)]; %#ok<AGROW>
end
fprintf('%d training images, %d test images, %d classes\n\n', ...
        size(Xtrain, 1), size(Xtest, 1), nClasses);

%% Classify every test image
nTest   = size(Xtest, 1);
predNN  = zeros(nTest, 1);
predKNN = zeros(nTest, 1);
for i = 1:nTest
    dist = sqrt(sum((Xtrain - Xtest(i, :)).^2, 2));   % Euclidean distance
    [~, order] = sort(dist);
    predNN(i)  = ytrain(order(1));                    % closest neighbour
    predKNN(i) = mode(ytrain(order(1:K)));            % majority vote of K
end

%% Results
fprintf('Overall accuracy\n');
fprintf('  NN       : %5.1f %%\n', 100 * mean(predNN  == ytest));
fprintf('  KNN (K=%d): %5.1f %%\n\n', K, 100 * mean(predKNN == ytest));

fprintf('%-12s %6s %8s %8s\n', 'Class', 'Tests', 'NN', 'KNN');
for c = 1:nClasses
    idx = ytest == c;
    fprintf('%-12s %6d %7.1f%% %7.1f%%\n', classes{c}, sum(idx), ...
            100 * mean(predNN(idx) == c), 100 * mean(predKNN(idx) == c));
end

% Confusion matrix for KNN (rows = true class, columns = predicted class)
confKNN = accumarray([ytest predKNN], 1, [nClasses nClasses]);
fprintf('\nKNN confusion matrix (rows = true, columns = predicted):\n');
disp(array2table(confKNN, 'VariableNames', classes, 'RowNames', classes));
