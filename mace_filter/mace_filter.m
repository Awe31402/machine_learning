clear;
close all;
clc;
filenames = dir('pattern.jp2');
num_files = numel(filenames);
img = imread(filenames(1).name);
[row, col] = size(img);

I = imread('target.jp2');
I = rgb2gray(I);
[targetRow, targetCol] = size(I);
targetRow = 2^nextpow2(targetRow + row - 1);
targetCol = 2^nextpow2(targetCol + col - 1);

for i = 1 : num_files
    printf('%s\n', filenames(i).name);
    img = imread(filenames(i).name);
    img = rgb2gray(img);
    img = double(img);
    img = fft2(img, targetRow, targetCol);
    X(:, i) = reshape((img), [(targetRow * targetCol) , 1]);
end

% Starting design MACE filter
avgps = mean(abs(X) .^ 2 , 2);
D = diag(avgps);
u = 10 ^ 6 * ones(num_files, 1);
D_inv = inv(D);
H = D_inv * X * inv(X' * D_inv * X) * u;
H_mace = reshape((H), [targetRow, targetCol]);

% Verify face recognution
I = double(I);
m = fft2(I, targetRow, targetCol);

k = m .* conj(H_mace);

g = ifft2(k);
figure;
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
