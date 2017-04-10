clear;
close all;

male_files = dir('m*.jpg');
female_files = dir('f*.jpg');

female_start = numel(male_files) + 1;

all_files = [male_files; female_files];
traning_data_num = numel(all_files);

labels = [zeros(1, female_start - 1), ones(1, traning_data_num - female_start + 1)];
labels = labels + ones(1, traning_data_num);
% generate all file matrix X
for i = 1 : traning_data_num
    img = imread(all_files(i).name);
    [row col] = size(img);
    img = double(img);
    X(:, i) = reshape(img, [(row * col), 1]);
end

eig_num = 24;

[eigenFaces pcaBasis] = pca(X, eig_num);

[fisherFaces ldaBasis] = lda(eigenFaces, labels);

labels = labels - ones(1, traning_data_num);
labels(1: female_start - 1) = -1 * ones(1, female_start - 1);
[w, b] = svm(fisherFaces, labels);

% Verify male or female by K-NN algorithm
tst_file = 'test.jpg';
nearest_num = 3;
img = imread(tst_file);
[row col] = size(img);
img = double(img);
reshape_img = reshape(img, [(row * col), 1]);
downDat = pcaBasis' * reshape_img;
%size(downDat)
optDat = ldaBasis' * downDat;
for i = 1: traning_data_num
    dist(i) = norm(optDat - fisherFaces(:, i));
end

female = 0;
male = 0;
max_dist = max(dist);
for i = 1: nearest_num
    index = find(dist == min(dist));
    if (index >= female_start)
        female++;
    else
        male++;
    end
    dist(index) = max_dist;
end

if (female > male)
    printf('K-NN: test data is female\n');
else
    printf('K-NN: test data is male\n');
end

% Verify Gender By SVM
class = w' * optDat + b;
if (class < 0)
    printf('SVM: test data is male\n');
else
    printf('SVM: test data is female\n');
end
