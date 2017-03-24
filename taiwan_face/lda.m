function [optimizedData, transisionMat] = lda(X, labels)
    %labels(i) = 1; -> male
    %labels(i) = 2; -> female
    global_mean = mean(X, 2);
    [dataDim dataNum] = size(X);

    classMatrix = [];
    Sb = zeros(dataDim, dataDim);
    Sw = zeros(dataDim, dataDim);
    for i = 1 : dataNum
        classMatrix(:, i, labels(i)) = X(:,i);
    end

    for i = 1 : max(labels)
        [r c] = size(classMatrix(:, :, i));
        diff_global = classMatrix(:, :, i) - repmat(global_mean,[1, c]);
        classMean = mean(classMatrix(:, :, i), 2);
        diff_inner = classMatrix(:, :, i) - repmat(classMean, [1, c]);
        Sb += diff_global * diff_global';
        Sw += diff_inner * diff_inner';
    end

    [eigv d] = eig(Sb, Sw);
    for i = 1 : dataDim
        eigv(:, i) = eigv(:, i) / norm(eigv(:, i));
    end
    optimizedData = eigv' * X;
    transisionMat = eigv;
end
