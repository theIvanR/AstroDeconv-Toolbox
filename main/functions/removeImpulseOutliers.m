function J = removeImpulseOutliers(I)

    med3 = medfilt2(I, [3 3], 'symmetric');
    r = I - med3;

    % local robust scale (instead of global MAD)
    localVar = medfilt2(abs(r), [7 7], 'symmetric');
    sigma = localVar + eps('single');

    % more aggressive thresholding
    k = 3.0;

    mask = abs(r) > k * sigma;

    % optional: expand mask slightly to catch small clusters
    mask = imdilate(mask, strel('square', 3));

    J = I;
    J(mask) = med3(mask);
end
