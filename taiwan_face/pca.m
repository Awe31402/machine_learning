clear;
clc;
close all;

male_files = dir('m*.jpg');
female_files = dir('f*.jpg');

female_start = numel(male_files) + 1;

all_files = [male_files; female_files];
traning_data_num = numel(all_files);

% generate all file matrix X
eig_num = 21;
for i = 1 : traning_data_num
    img = imread(all_files(i).name);
    [row col] = size(img);
    img = double(img);
    X(:, i) = reshape(img, [(row * col), 1]);
end

% Start PCA FLOW....
Xm = mean(X, 2);
X = X - repmat(Xm, [1, traning_data_num]);

sigma_s = X' * X;
[v, d] = eigs(sigma_s, eig_num);

w = X * v;

[w_row w_col] = size(w);

for i = 1 : w_col
    w(:, i) = w(:, i) / norm(w(:, i));
end

inner_proc = w' * X;
% END PCA FLOW....

for j = 1 : traning_data_num
    recon_img = Xm;
    for i = 1: eig_num
        recon_img += inner_proc(i, j) * w(: , i);
    end

    recon_img = reshape(recon_img, [row , col]);
    figure;
    imshow(mat2gray(recon_img));
end

for j = 1 : eig_num / 3
    ind = (j - 1) * 3;
    figure;
    for i = 1 : traning_data_num
        if (i < female_start)
            plot3(inner_proc(ind + 1,i), inner_proc(ind + 2, i), inner_proc(ind + 3,i), 'bx')
        else
            plot3(inner_proc(ind + 1,i), inner_proc(ind + 2, i), inner_proc(ind + 3,i), 'ro')
        end
        hold on;
    end
    grid on;
end
