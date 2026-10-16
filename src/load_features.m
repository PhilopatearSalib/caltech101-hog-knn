function F = load_features(matFile)
%LOAD_FEATURES Load a HOG feature file and return an N-by-1080 matrix.
%   F = LOAD_FEATURES(matFile) returns one row per image.
%
%   The 2019 feature files are not consistent, so they are not loaded by
%   variable name:
%     - Some files store their features under the wrong variable name
%       (e.g. laptop_featurevector.mat contains "sunflower_featurevector").
%       Loading them straight into the workspace silently overwrote the
%       real sunflower features.
%     - The test files contain all-zero placeholder slices before the real
%       test images (the extraction loop started at k = 46 or 51 and wrote
%       into features(:,:,k)).
%
%   This function loads the file into a struct, takes the 1-by-D-by-N
%   feature array whatever it is called, and drops all-zero rows.

S = load(matFile);
names = fieldnames(S);
F = [];
for i = 1:numel(names)
    v = S.(names{i});
    if isnumeric(v) && ndims(v) == 3 && size(v, 1) == 1
        F = permute(v, [3 2 1]);   % 1 x D x N  ->  N x D
        break
    end
end

if isempty(F)
    error('load_features:notFound', ...
          'No 1-by-D-by-N feature array found in %s', matFile);
end

F = double(F(any(F ~= 0, 2), :));
end
