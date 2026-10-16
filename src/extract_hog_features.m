% EXTRACT_HOG_FEATURES  Rebuild the feature files in ../features from the
% Caltech-101 images.
%
% Not needed to run the classifier: the 2019 feature files are already in
% ../features. Use this only to regenerate them. It writes files in the
% same format (variable "featurevector", size 1 x 1080 x N) but without
% the zero placeholder slices, and with a plain 70/30 split per class.
% Results will differ slightly from the published numbers, which come
% from the 2019 split.
%
% Expects the dataset at ../data/101_ObjectCategories (see README).
% Requires the Computer Vision Toolbox (extractHOGFeatures).

clc; clear;

datasetDir = fullfile('..', 'data', '101_ObjectCategories');
outDir     = fullfile('..', 'features');
imageSize  = [60 50];   % rows x cols, as in 2019 -> 1080 HOG features
trainRatio = 0.7;

% Caltech-101 folder name, short name used in the feature file names
classes = {
    'sunflower',  'sunflower'
    'crab',       'crab'
    'chair',      'chair'
    'cellphone',  'cellphone'
    'dolphin',    'dolphin'
    'stop_sign',  'stopsign'
    'helicopter', 'helicopter'
    'lamp',       'lamp'
    'kangaroo',   'kangaroo'
    'laptop',     'laptop'
};

for c = 1:size(classes, 1)
    listing = dir(fullfile(datasetDir, classes{c, 1}, '*.jpg'));
    files   = sort({listing.name});
    n       = numel(files);
    nTrain  = round(trainRatio * n);

    feats = [];
    for k = 1:n
        img = imread(fullfile(datasetDir, classes{c, 1}, files{k}));
        img = imresize(img, imageSize);
        feats(k, :) = extractHOGFeatures(img); %#ok<SAGROW>
    end

    featurevector = permute(feats(1:nTrain, :), [3 2 1]);   % 1 x D x N
    save(fullfile(outDir, [classes{c, 2} '_featurevector.mat']), 'featurevector');

    featurevector = permute(feats(nTrain+1:end, :), [3 2 1]);
    save(fullfile(outDir, [classes{c, 2} 'Test_featurevector.mat']), 'featurevector');

    fprintf('%-12s %3d train, %3d test\n', classes{c, 1}, nTrain, n - nTrain);
end
