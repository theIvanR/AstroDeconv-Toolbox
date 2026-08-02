% Note, not commutative, so: Denoise FIRST, Flat-divide SECOND, Deconvolve THIRD.

%% 0: ensure that all the files were converted to tiff via auxilliary script

%% 1: Construct jobs and execute
addpath(genpath(pwd));
rootDir = 'data';
jobs = buildJobList(rootDir, '.tiff');

% 1.1 hard check for light frames
if isempty(jobs.light); error('No LIGHT frames found.');
end

% 1.2 soft check for flat frames
if isempty(jobs.flat)
    warning('No FLAT frames found. Skipping flat correction.');
else
    masterFlat = make_master_flat(jobs.flat);
end

%% 2: Analyze
net = denoisingNetwork("DnCNN");

outDir = fullfile(pwd, 'output');
if ~exist(outDir, 'dir'); mkdir(outDir);
end

for i = 1:numel(jobs.light)
    inFile = jobs.light(i).filePath;
    [~, baseName, ~] = fileparts(jobs.light(i).name);
    outFile = fullfile(outDir, [baseName '.tiff']);

    fprintf('Processing %s (%d/%d)\n', jobs.light(i).name, i, numel(jobs.light));

    I = im2single(imread(inFile));
    if ndims(I) > 3; error('Image has more than 3 dimensions: %s', inFile); end
    if size(I,3) > 4; error('Image has %d channels; hard limit is 4.', size(I,3));  end
    if ndims(I) == 2; I = reshape(I, size(I,1), size(I,2), 1); end


    O = zeros(size(I), 'like', I);

    % 1A Normalize per Channel
    parfor c = 1:size(I,3)
        chan = I(:,:,c);

        % 1: Remove noise per channel
        chan = removeImpulseOutliers(chan);

        % 2: Flat divide
        if ~isempty(masterFlat)
            F = getFlatChannel(masterFlat, c);
            F = max(F, eps('single'));
            chan = chan ./ F;
        end

        % write to channel
        O(:,:,c) = chan;
    end

    % Remove color outliers in O
    O = removeColorImpulseOutliers(O);
    
    % 1B Denoise per Channel (NOW correctly using O)
    for c = 1:size(I,3)
        chan = O(:,:,c);   % ✅ FIX

        chan = sqrt(chan + eps('single'));
        chan = denoiseImage(chan, net);
        chan = max(chan.^2, 0);

        O(:,:,c) = chan;
    end


    % 2: Estimate stacked PSF from average channel image and deconvolve
    A = mean(O, 3);

    [psfKernel, psfStats] = estimateStackedStarPSF(A, 80, 50);

    fprintf('stars=%d/%d | FWHM=%.2f px | ellip=%.3f | radial=[%.3g %.3g %.3g] | coma=[%.3g %.3g]\n', ...
        psfStats.nUsed, psfStats.nSelected, ...
        psfStats.fwhmMajor, psfStats.ellipticity, ...
        psfStats.psfModel.coeff(1), psfStats.psfModel.coeff(2), psfStats.psfModel.coeff(3), ...
        psfStats.psfModel.coma.coeff(1), psfStats.psfModel.coma.coeff(2));

    parfor c = 1:size(O,3)
        O(:,:,c) = deconvolveConstantPSF(O(:,:,c), psfKernel, 5);
    end
     
    % Save
    if size(O,3) == 1; O = O(:,:,1);
    end
    imwrite(im2uint16(O), outFile, 'Compression', 'lzw');
end


%% Helpers Variant PSF
function O = deconvolveConstantPSF(I, psf, nIter)

    I = im2single(I);
    psf = im2single(psf);

    psf(psf < 0) = 0;
    psf = psf / max(sum(psf(:)), eps('single'));

    O = deconvlucy(I, psf, nIter);
end

% % optical transfer manifold learning via higher order radial corrections
function [psfKernel, stats] = estimateStackedStarPSF(I, nKeep, minSep)

    if nargin < 2 || isempty(nKeep)
        nKeep = 80;
    end
    if nargin < 3 || isempty(minSep)
        minSep = 16;
    end

    I = im2single(I);
    [H, W] = size(I);

    cx0 = (W + 1) / 2;
    cy0 = (H + 1) / 2;

    bg = medfilt2(I, [31 31], 'symmetric');
    J = I - bg;
    J(J < 0) = 0;
    J = imgaussfilt(J, 1.0);

    thr = median(J(:)) + 6 * mad(J(:), 1);
    peaks = imregionalmax(J) & (J > thr);
    [y0, x0] = find(peaks);

    stats.nSelected = numel(x0);
    if isempty(x0)
        error('No candidate stars found.');
    end

    peakScore = J(sub2ind(size(J), y0, x0));
    [~, ord] = sort(peakScore, 'descend');
    x0 = x0(ord);
    y0 = y0(ord);

    keep = false(size(x0));
    blocked = false(H, W);

    halfSep = max(1, ceil(minSep / 2));
    for ii = 1:numel(x0)
        x = x0(ii);
        y = y0(ii);

        if ~blocked(y, x)
            keep(ii) = true;

            y1 = max(1, y - halfSep);
            y2 = min(H, y + halfSep);
            x1 = max(1, x - halfSep);
            x2 = min(W, x + halfSep);

            blocked(y1:y2, x1:x2) = true;
        end
    end

    x0 = x0(keep);
    y0 = y0(keep);
    peakScore = peakScore(keep);

    r = 8;
    sz = 2*r + 1;
    target = r + 1;

    nCand = numel(x0);

    stamps = zeros(sz, sz, nCand, 'single');
    fluxes = zeros(nCand, 1, 'single');
    fwhmMajor = zeros(nCand, 1, 'single');
    fwhmMinor = zeros(nCand, 1, 'single');
    ell = zeros(nCand, 1, 'single');
    theta = zeros(nCand, 1, 'single');
    xs = zeros(nCand, 1, 'single');
    ys = zeros(nCand, 1, 'single');
    comaObs = zeros(nCand, 1, 'single');

    nUsedStars = 0;

    [X, Y] = meshgrid(1:sz, 1:sz);
    localX = X - target;
    localY = Y - target;

    for i = 1:nCand

        x = x0(i);
        y = y0(i);

        if x-r < 1 || y-r < 1 || x+r > W || y+r > H
            continue;
        end

        stamp = J(y-r:y+r, x-r:x+r);

        border = [stamp(1,:), stamp(end,:), stamp(:,1).', stamp(:,end).'];
        bkg = median(border);

        S = stamp - bkg;
        S(S < 0) = 0;

        flux = sum(S(:));
        if flux <= 0
            continue;
        end

        Wt = S / flux;

        cx = sum(X(:) .* Wt(:));
        cy = sum(Y(:) .* Wt(:));

        dx = X - cx;
        dy = Y - cy;

        Cxx = sum(Wt(:) .* dx(:).^2);
        Cyy = sum(Wt(:) .* dy(:).^2);
        Cxy = sum(Wt(:) .* dx(:) .* dy(:));

        C = [Cxx, Cxy; Cxy, Cyy];

        [V, D] = eig(C);
        sig = sqrt(max(diag(D), eps('single')));
        [sig, idxEig] = sort(sig, 'descend');
        v1 = V(:, idxEig(1));

        elli = 1 - sig(2) / max(sig(1), eps('single'));

        if elli > 0.80
            continue;
        end

        dxShift = target - cx;
        dyShift = target - cy;

        S2 = shiftStampLinear(S, dxShift, dyShift);
        S2(S2 < 0) = 0;

        s2sum = sum(S2(:));
        if s2sum <= 0
            continue;
        end

        S2 = S2 / s2sum;

        nUsedStars = nUsedStars + 1;

        stamps(:,:,nUsedStars) = S2;
        fluxes(nUsedStars) = flux;
        fwhmMajor(nUsedStars) = 2 * sqrt(2*log(2)) * sig(1);
        fwhmMinor(nUsedStars) = 2 * sqrt(2*log(2)) * sig(2);
        ell(nUsedStars) = elli;
        theta(nUsedStars) = atan2(v1(2), v1(1));
        xs(nUsedStars) = x;
        ys(nUsedStars) = y;

        % coma observable (stable odd moment proxy)
        rr = hypot(x - cx0, y - cy0);
        if rr > 0
            ux = (x - cx0) / rr;
            uy = (y - cy0) / rr;
            s = localX * ux + localY * uy;
            comaObs(nUsedStars) = sum(S2(:) .* (s(:).^3));
        else
            comaObs(nUsedStars) = 0;
        end
    end

    if nUsedStars == 0
        error('No usable stars remained after filtering.');
    end

    stamps = stamps(:,:,1:nUsedStars);
    fluxes = fluxes(1:nUsedStars);
    fwhmMajor = fwhmMajor(1:nUsedStars);
    fwhmMinor = fwhmMinor(1:nUsedStars);
    ell = ell(1:nUsedStars);
    theta = theta(1:nUsedStars);
    xs = xs(1:nUsedStars);
    ys = ys(1:nUsedStars);
    comaObs = comaObs(1:nUsedStars);

    [~, ord2] = sort(fluxes, 'descend');
    nUsed = min(nKeep, numel(ord2));
    ord2 = ord2(1:nUsed);

    P = mean(stamps(:,:,ord2), 3);
    P = max(P, 0);
    P = P / sum(P(:));

    stats = psfMoments(P);
    stats.nSelected = numel(peakScore);
    stats.nUsed = nUsed;

    % radial model
    rField = hypot(xs - cx0, ys - cy0);
    rNorm = rField ./ max(rField + eps('single'));

    B = [ones(size(rNorm)), rNorm, rNorm.^2];
    lambda = 1e-3;
    coeff = (B' * B + lambda * eye(3,'single')) \ (B' * fwhmMajor);

    stats.psfModel.type = "radial_quadratic";
    stats.psfModel.coeff = coeff;

    % ===== COMA FIT (NO GAIN, FULLY LEARNED) =====

    phi = atan2(ys - cy0, xs - cx0);
    valid = rNorm > 0.15;

    if nnz(valid) >= 6
        Cb = [rNorm(valid).^2 .* cos(phi(valid)), ...
              rNorm(valid).^2 .* sin(phi(valid))];

        y = comaObs(valid);

        lambdaC = 1e-3;
        comaCoeff = (Cb' * Cb + lambdaC * eye(2,'single')) \ (Cb' * y);
    else
        comaCoeff = zeros(2,1,'single');
    end

    stats.psfModel.coma.coeff = comaCoeff;

    % APPLY COMA DIRECTLY TO KERNEL
    if norm(comaCoeff) > 0
        P = applyKernelComa(P, comaCoeff);
    end

    psfKernel = P;
end

function P2 = applyKernelComa(P, c)

    ax = c(1);
    ay = c(2);

    P = max(P, 0);
    P = P / sum(P(:));

    [H,W] = size(P);
    [X,Y] = meshgrid(1:W, 1:H);

    cx = (W+1)/2;
    cy = (H+1)/2;

    dx = X - cx;
    dy = Y - cy;

    r = hypot(dx, dy);
    rn = r ./ max(r(:) + eps('single'));

    % FULLY LEARNED (no gain)
    u = ax * rn.^2;
    v = ay * rn.^2;

    Xq = X + u;
    Yq = Y + v;

    P2 = interp2(X, Y, P, Xq, Yq, 'linear', 0);

    P2 = max(P2, 0);
    P2 = P2 / sum(P2(:));
end

function stats = psfMoments(P)
    P = im2single(P);
    P(P < 0) = 0;
    P = P / max(sum(P(:)), eps('single'));

    [H, W] = size(P);
    [X, Y] = meshgrid(1:W, 1:H);

    cx = sum(X(:) .* P(:));
    cy = sum(Y(:) .* P(:));

    dx = X - cx;
    dy = Y - cy;

    Cxx = sum(P(:) .* dx(:).^2);
    Cyy = sum(P(:) .* dy(:).^2);
    Cxy = sum(P(:) .* dx(:) .* dy(:));

    C = [Cxx, Cxy; Cxy, Cyy];
    [V, D] = eig(C);

    sig = sqrt(max(diag(D), eps('single')));
    [sig, idx] = sort(sig, 'descend');
    v1 = V(:, idx(1));

    stats.fwhmMajor = 2 * sqrt(2 * log(2)) * sig(1);
    stats.fwhmMinor = 2 * sqrt(2 * log(2)) * sig(2);
    stats.ellipticity = 1 - sig(2) / max(sig(1), eps('single'));
    stats.theta = atan2(v1(2), v1(1));
end

function T = shiftStampLinear(S, dx, dy)
    [H, W] = size(S);
    [X, Y] = meshgrid(1:W, 1:H);
    T = interp2(X, Y, S, X - dx, Y - dy, 'linear', 0);
end






