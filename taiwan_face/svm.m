function [w, b] = svm(data, labels)
    [dimension, dataNum] = size(data);
    markedData = repmat(labels, [dimension, 1]) .* data;
    K = markedData' * markedData;
    q = -1 * ones(dataNum, 1);

    % Quadratic Programming for KKT dual
    [alpha, OBJS, INFO, LAMBDA] = qp([], K, q, [], [],
        zeros(dataNum, 1), []);

    %Check contraint
    %alpha' * labels'
    weight = repmat(alpha', [dimension, 1]);
    w = sum(weight .* markedData, 2);

    negative_index = find(labels == -1);
    positive_index = find(labels == 1);

    no_bias = w' * data;
    negative_max = max(no_bias(negative_index));
    positive_min = min(no_bias(positive_index));
    b = -1 * (positive_min + negative_max) / 2;
