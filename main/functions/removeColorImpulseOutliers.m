function J = removeColorImpulseOutliers(I)

    I = im2single(I);

    if size(I,3) < 3
        J = I;
        return;
    end

    R = I(:,:,1);
    G = I(:,:,2);
    B = I(:,:,3);

    % ------------------------------------------------------------
    % 1. Build local luminance structure (robust reference field)
    % ------------------------------------------------------------
    Y = 0.299*R + 0.587*G + 0.114*B;

    Ymed = medfilt2(Y, [3 3], 'symmetric');

    % ------------------------------------------------------------
    % 2. Channel deviation from local structure
    % ------------------------------------------------------------
    dR = abs(R - Ymed);
    dG = abs(G - Ymed);
    dB = abs(B - Ymed);

    % ------------------------------------------------------------
    % 3. Robust scale estimate (spatially adaptive)
    % ------------------------------------------------------------
    scale = medfilt2(dR + dG + dB, [7 7], 'symmetric');
    scale = scale + eps('single');

    % ------------------------------------------------------------
    % 4. Adaptive threshold (NOT fixed k)
    % ------------------------------------------------------------
    k = 4.0;

    mask = (dR > k*scale) | (dG > k*scale) | (dB > k*scale);

    % ------------------------------------------------------------
    % 5. Morphological cleanup (avoid single-pixel artifacts)
    % ------------------------------------------------------------
    mask = imdilate(mask, strel('square', 3));

    % ------------------------------------------------------------
    % 6. Replacement strategy (key improvement)
    %    → collapse to local luminance, not per-channel median
    % ------------------------------------------------------------
    R(mask) = Ymed(mask);
    G(mask) = Ymed(mask);
    B(mask) = Ymed(mask);

    J = cat(3, R, G, B);
end