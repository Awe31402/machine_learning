clear;
close all;
clc;
filenames = dir('pattern.jp2');
num_files = numel(filenames);
img = imread(filenames(1).name);
[row, col] = size(img);

I = imread('target.jp2');
%I = rgb2gray(I);
[targetRow, targetCol] = size(I);
targetRow = 2^nextpow2(targetRow + row - 1);
targetCol = 2^nextpow2(targetCol + col - 1);

%figure;
for i = 1 : num_files
    printf('%s\n', filenames(i).name);
    img = imread(filenames(i).name);
    %subplot(5 ,4, i);
    %imshow(img);
    img = rgb2gray(img);
    img = double(img);
    img = fft2(img, targetRow, targetCol);
    X(:, i) = reshape(transpose(img), [(targetRow * targetCol) , 1]);
end

% Starting design MACE filter
avgps = mean(abs(X) .^ 2 , 2);
D = diag(avgps);
u = 10 ^ 6 * ones(num_files, 1);
D_inv = inv(D);
H = D_inv * X * inv(X' * D_inv * X) * u;
H_mace = transpose(reshape(transpose(H), [targetCol, targetRow]));
%H_tmp = zeros(targetRow, targetCol);
%H_tmp(1: row, 1:col) = H_mace(1: row, 1:col);
%H_mace = H_tmp;

% Verify face recognution
I = double(I);
m = fft2(I, targetRow, targetCol);
%H_mace_p = zeros(size(m));
%H_mace_p(1: row, 1 : col) = H_mace;
k = m .* conj(H_mace);
%figure;
%mesh(abs(k));
g = ifft2(k);
figure;
%begR = floor((targetRow - size(I)(1)) /2);
%begC = floor((targetCol - size(I)(2)) /2);
%endR = floor((targetRow + size(I)(1)) /2);
%endC = floor((targetCol + size(I)(2)) /2);
%g = abs(g(begR : endR, begC:endC));
g = abs(g(1:size(I)(1), 1:size(I)(2)));
mesh(g);

figure;
imshow(mat2gray(I));
hold on;
[i,j] = find(g == max(max(g)))
g(i,j) = 0;
plot(j + 10 , i + 10, 'ro');
[i,j] = find(g == max(max(g)))
plot(j + 10, i + 10, 'ro');
g(i,j) = 0;
[i,j] = find(g == max(max(g)))
plot(j + 10, i + 10, 'ro');

figure;
imagesc(g);
