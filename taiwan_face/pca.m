function [downGradeData, eigenVectors] = pca(X, dimension)
    % Start PCA FLOW....
    Xm = mean(X, 2);

    [imageDimension traning_data_num] = size(X);

    X_orig = X;

    X = X - repmat(Xm, [1, traning_data_num]);

    eig_num = dimension;

    sigma_s = X' * X;
    [v, d] = eigs(sigma_s, eig_num);

    w = X * v;

    [w_row w_col] = size(w);

    for i = 1 : w_col
        w(:, i) = w(:, i) / norm(w(:, i));
    end

    inner_proc = w' * X_orig;
    downGradeData = inner_proc;
    eigenVectors = w;
end
% END PCA FLOW....

%for j = 1 : traning_data_num
%    recon_img = Xm;
%    for i = 1: eig_num
%        recon_img += inner_proc(i, j) * w(: , i);
%    end
%
%    recon_img = reshape(recon_img, [row , col]);
%    figure;
%    imshow(mat2gray(recon_img));
%end
