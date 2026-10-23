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
    maskR = dR > k*scale;
    maskG = dG > k*scale;
    maskB = dB > k*scale;

    % ------------------------------------------------------------
    % 5. Replace only the offending channel using its local median.
    %    Preserve the other channels and their color information.
    % ------------------------------------------------------------
    Rmed = medfilt2(R, [3 3], 'symmetric');
    Gmed = medfilt2(G, [3 3], 'symmetric');
    Bmed = medfilt2(B, [3 3], 'symmetric');

    R(maskR) = Rmed(maskR);
    G(maskG) = Gmed(maskG);
    B(maskB) = Bmed(maskB);

    J = cat(3, R, G, B);

end