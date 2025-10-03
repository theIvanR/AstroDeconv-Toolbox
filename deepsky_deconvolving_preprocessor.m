%% process_arw_modular.m
% Single-file pipeline:
%  - scans Input/*.arw, Flat/*.arw
%  - converts each RAW to an RGB linear image using rawToRGB()
%  - optional IR handling, AutoWB, denoise, lowpass (optional)
%  - saves TIFF outputs to Output/

clear; clc; close all; 

%% Global I/O
srcDir  = 'Input'; if ~exist(srcDir, 'dir'),  mkdir(srcDir);  end
flatDir = 'Flat';if ~exist(flatDir, 'dir'), mkdir(flatDir); end
destDir = 'Output';if ~exist(destDir, 'dir'), mkdir(destDir); end

% Collect files
rawFiles  = dir(fullfile(srcDir,  '*.arw'));
flatFiles = dir(fullfile(flatDir, '*.arw'));


%% Global Options
blackLevel = [512 512 512 512];       % [R, G1, G2, B]
useAutoWB = true;                     % true to perform white balance
doHampel = true; % Hampel settings (applied on black-subtracted raw, removes hot pixels 8 average kernel) 
        
% CNN Denoiser (DnCNN)  
net = denoisingNetwork("DnCNN");

%% ---------------------- Process Flat Frames (Production-quality) ----------------------
tic;
fprintf('Processing %d flat frames...\n', numel(flatFiles));

% --------------------------- Tunable parameters ---------------------------
params.percentileScale        = 99;    % percentile mapped to 1.0 when normalizing channel
params.minNormVal             = 0.002; % floor for normalized flat pixels (avoids enormous inverse)
params.maxGain                = 100;   % hard cap for inverse gain
params.doSmooth               = true;  % smooth master mean before SNR calc
params.smoothSigmaPx          = 15;    % gaussian sigma in px for smoothing (choose per use-case)
params.snrPercentileClipping  = 95;    % percentile used to compute snrMax
params.snrFloor               = 0.30;  % below this SNR -> near-zero confidence

% Radial taper (toned down), use for low snr flats
params.doRadialTaper = true;
params.innerR = 0.50;   % fraction (0..1) of radius for full correction
params.outerR = 0.99;   % fraction (0..1) where we blend fully back to unity
params.alpha  = 0.7;   % taper strength (0..1)
params.power  = 0.8;    % shape exponent (<1 gentler, >1 steeper)
params.useSigmoid = false;

% Diagnostics
params.doDiagnostics = false;
diagDir = fullfile(pwd, 'flat_diag');

% -------------------------- Validate inputs & init --------------------------
if ~exist('flatFiles','var') || isempty(flatFiles)
    masterFlat  = [];
    masterFlatInv = [];
    fprintf('No flat frames found — skipping flat creation.\n\n');
    return;
end

% quick type & shape checks (assume rawToRGB returns HxWx3)
nFlats = numel(flatFiles);
fprintf('Found %d flat files. Loading and stacking...\n', nFlats);

% Preallocate stack lazily after first successful read to know image size
flatStack = [];
loaded = 0;

for fidx = 1:nFlats
    rawPath = fullfile(flatFiles(fidx).folder, flatFiles(fidx).name);
    fprintf('  [%d/%d] Loading: %s\n', fidx, nFlats, flatFiles(fidx).name);
    try
        flatRGB = rawToRGB(rawPath, 'BlackLevel', blackLevel, 'RemoveHotPixels', true);
    catch ME
        warning('  Failed to read flat "%s": %s — skipping this file.', flatFiles(fidx).name, ME.message);
        continue;
    end

    % Validate shape & cast to single
    if ndims(flatRGB) ~= 3 || size(flatRGB,3) ~= 3
        warning('  Unexpected flat image shape for "%s" — skipping.', flatFiles(fidx).name);
        continue;
    end
    flatRGB = single(flatRGB);

    if loaded == 0
        [H, W, ~] = size(flatRGB);
        flatStack = zeros(H, W, 3, nFlats, 'single');  % max allocation; we track actual count with 'loaded'
    end

    loaded = loaded + 1;
    flatStack(:,:,:,loaded) = flatRGB;
end

if loaded == 0
    masterFlat = [];
    masterFlatInv = [];
    warning('No usable flat frames were loaded.');
    return;
end

% Trim unused preallocated slots if any
if loaded < nFlats
    flatStack(:,:,:,loaded+1:end) = [];
end

% ---------------------- Compute per-pixel statistics -----------------------
% Per-pixel mean and std across the stack
masterMean = mean(flatStack, 4);   % H x W x 3
masterStd  = std(flatStack, 0, 4); % H x W x 3

masterMean = single(masterMean);
masterStd  = single(masterStd);

% ------------------------- Optional smoothing ------------------------------
if params.doSmooth && params.smoothSigmaPx > 0
    % Use imgaussfilt with automatic filter sizing (safer than forcing an
    % odd size). Slightly more control then before
    for c = 1:3
        masterMean(:,:,c) = imgaussfilt(masterMean(:,:,c), params.smoothSigmaPx);
    end
end

% ----------------------------- SNR -> confidence ---------------------------
tiny = 1e-12;
snrMap = masterMean ./ (masterStd + tiny);  % per-pixel SNR map

% determine snrMax robustly using percentile across all valid SNR entries
snrVals = snrMap(:);
validMask = isfinite(snrVals) & (snrVals > 0);
if any(validMask)
    snrMax = prctile(snrVals(validMask), params.snrPercentileClipping);
else
    snrMax = max(snrMap(:));
end
if isempty(snrMax) || snrMax <= 0
    snrMax = max(snrMap(:));
end
snrMax = max(single(snrMax), tiny);

% Map SNR to confidence [0,1] per channel (linear ramp between snrFloor..snrMax)
confMap = zeros(size(snrMap), 'single');
for c = 1:3
    conf = (snrMap(:,:,c) - params.snrFloor) ./ (snrMax - params.snrFloor);
    confMap(:,:,c) = min(max(conf, 0), 1);
end

% ------------------------ Normalize master mean -> masterFlat --------------
masterFlat = zeros(size(masterMean), 'single');
for c = 1:3
    ch = masterMean(:,:,c);
    pos = ch(ch > 0);
    if isempty(pos)
        warning('Flat channel %d: no positive pixels found; using unity channel.', c);
        masterFlat(:,:,c) = ones(size(ch), 'single');
        continue;
    end

    pval = prctile(pos, params.percentileScale);
    if pval <= tiny
        pval = median(pos);
        if pval <= tiny
            warning('Flat channel %d: percentile and median are tiny — using unity.', c);
            masterFlat(:,:,c) = ones(size(ch), 'single');
            continue;
        end
    end

    normCh = ch / pval;                       % representative bright pixels ~ 1.0
    normCh = max(normCh, params.minNormVal);  % floor tiny pixels to avoid extreme inverse
    masterFlat(:,:,c) = single(normCh);
    fprintf(' Flat ch%d: p%g = %.6g (snrFloor=%.3g snrMax~%.3g)\n', c, params.percentileScale, pval, params.snrFloor, snrMax);
end

% ------------------------- Compute final inverse gain ---------------------
rawInv = 1 ./ masterFlat;                    % per-pixel raw inverse
masterFlatInv = 1 + (rawInv - 1) .* confMap; % blend: 1 = no-op, rawInv = full correction
masterFlatInv = single(masterFlatInv);

% Hard cap gains
masterFlatInv(masterFlatInv > params.maxGain) = params.maxGain;

% ------------------------- Tonal / radial taper ----------------------------
if params.doRadialTaper
    innerR = params.innerR;
    outerR = params.outerR;
    alpha  = params.alpha;
    power  = params.power;
    useSigmoid = params.useSigmoid;

    % ensure radii valid
    if outerR <= innerR
        outerR = min(0.995, innerR + 0.01);
    end

    [H, W, ~] = size(masterFlatInv);
    [xg, yg] = meshgrid((1:W) - (W + 1)/2, (1:H) - (H + 1)/2);
    rnorm = sqrt(xg.^2 + yg.^2) / sqrt(((W+1)/2)^2 + ((H+1)/2)^2);
    t = (rnorm - innerR) ./ (outerR - innerR);
    t = min(max(t,0),1);

    if useSigmoid
        k = 8 * power;
        s = 1 ./ (1 + exp(-k * (t - 0.5)));
        blend_base = 1 - s;
        blend_base = (blend_base - min(blend_base(:))) / (max(blend_base(:)) - min(blend_base(:)) + eps);
    else
        if power == 1
            blend_base = 0.5 * (1 + cos(pi * t));
        else
            t_warp = t .^ power;
            blend_base = 0.5 * (1 + cos(pi * t_warp));
        end
    end

    % final blend: mix toward unity a bit using alpha
    final_blend = (1 - alpha) + alpha * blend_base;  % range [1-alpha .. 1]

    % apply same blend across channels (simple and robust)
    for c = 1:3
        b = final_blend;
        masterFlatInv(:,:,c) = 1 .* (1 - b) + masterFlatInv(:,:,c) .* b;
    end

    fprintf('Applied radial taper (inner=%.2f outer=%.2f alpha=%.2f power=%.2f sigmoid=%d)\n', ...
        innerR, outerR, alpha, power, useSigmoid);
end

% ------------------------------- Diagnostics --------------------------------
if params.doDiagnostics
    if ~exist(diagDir,'dir'), mkdir(diagDir); end
    imwrite(mat2gray(masterFlat(:,:,2)), fullfile(diagDir,'masterFlat_G.png'));
    imwrite(mat2gray(masterFlatInv(:,:,2)), fullfile(diagDir,'masterFlatInv_G.png'));
    imwrite(mat2gray(confMap(:,:,2)), fullfile(diagDir,'confidence_G.png'));
    save(fullfile(diagDir,'flat_maps.mat'),'masterMean','masterStd','masterFlat','masterFlatInv','confMap','-v7.3');
    fprintf('Diagnostics saved to %s\n', diagDir);
end

fprintf('Master flat ready: size=%dx%dx3  inv_min=%.3g inv_max=%.3g (maxGain=%.3g)\n', ...
    size(masterFlat,1), size(masterFlat,2), min(masterFlatInv(:)), max(masterFlatInv(:)), params.maxGain);

elapsedTime = toc;
fprintf('Flats pass done. Elapsed time: %.2f sec\n\n', elapsedTime);


%% Process Light Frames    
tic
fprintf('Processing %d Light frames...\n', length(rawFiles));

parfor idx = 1:length(rawFiles)
    rawFilePath = fullfile(rawFiles(idx).folder, rawFiles(idx).name);
    fprintf('\nProcessing %s (%d of %d)...\n', rawFiles(idx).name, idx, length(rawFiles));

    % Convert RAW -> linear RGB (single). rawToRGB handles hot pixel removal,
    % black subtraction and demosaic for RGGB pattern by default.
    rgbSignal = rawToRGB(rawFilePath, 'BlackLevel', blackLevel, 'RemoveHotPixels', true);

    % Apply flat Correction
    rgbSignal = rgbSignal .* masterFlatInv;

    % ----------------- AutoWB with selectable reference channel and IR -----------------
    refChannel = 'G';      % 'R', 'G', or 'B'  (reference channel whose median stays the same)
    allowAmplify = true;   % true = allow gains > 1; false = never amplify (only attenuate)

    if useAutoWB
        % build bright-pixel mask (robust)
        grayLevels = sum(rgbSignal, 3);
        threshold = prctile(single(grayLevels(:)), 99.5);
        mask = grayLevels >= threshold;
        if ~any(mask(:))
            threshold = prctile(single(grayLevels(:)), 99.0);
            mask = grayLevels >= threshold;
        end
        if ~any(mask(:))
            mask = true(size(grayLevels));
        end

        % extract channels
        R = rgbSignal(:,:,1); G = rgbSignal(:,:,2); B = rgbSignal(:,:,3);

        % compute medians on the mask
        scaleR = median(single(R(mask)));
        scaleG = median(single(G(mask)));
        scaleB = median(single(B(mask)));

        % defend against degenerate medians
        tiny = 1e-12;
        if ~isfinite(scaleR) || scaleR <= tiny, scaleR = median(single(R(:))) + tiny; end
        if ~isfinite(scaleG) || scaleG <= tiny, scaleG = median(single(G(:))) + tiny; end
        if ~isfinite(scaleB) || scaleB <= tiny, scaleB = median(single(B(:))) + tiny; end

        % choose reference scale
        switch upper(refChannel)
            case 'R', scaleRef = scaleR;
            case 'G', scaleRef = scaleG;
            case 'B', scaleRef = scaleB;
            otherwise
                warning('Unknown refChannel "%s", using G', refChannel);
                scaleRef = scaleG;
        end

        % compute gains relative to reference (ref gain == 1)
        gR = scaleRef / scaleR;
        gG = scaleRef / scaleG;
        gB = scaleRef / scaleB;

        % optionally prevent amplification (only allow attenuation)
        if ~allowAmplify
            gR = min(gR, 1);
            gG = min(gG, 1);
            gB = min(gB, 1);
        end

        % apply gains
        rgbSignal(:,:,1) = R .* gR;
        rgbSignal(:,:,2) = G .* gG;
        rgbSignal(:,:,3) = B .* gB;

        % diagnostics
        fprintf('AutoWB(ref=%s): nnz(mask)=%d raw_scales=[%.6g %.6g %.6g] gains=[%.6g %.6g %.6g] WB_Max=%.6g\n', ...
            refChannel, nnz(mask), scaleR, scaleG, scaleB, gR, gG, gB, max(rgbSignal(:)));

    end

    % ----------------- Apply Denoising and Image Processing Methids Here -----------------

    % A 1: Lowpass and resize
    rgbSignal = applyGaussianLowPass(rgbSignal, 3); %minPixelDetail = 3 pixel
    rgbSignal = imresize(rgbSignal, 0.5, 'lanczos3');

    % A 2: Normalize to [0, 1] after resize
    fprintf('Normalized to [0, 1]\n')
    prescaler = max(rgbSignal(:));
    rgbSignal = rgbSignal / prescaler;
    rgbSignal(rgbSignal < 0) = 0;
    

    % B: Generate PSF

    % 1: Naive PSF Guess
    % fwhm = 6;
    % sigma = fwhm / 2.3548;
    % psfSize = ceil(fwhm*2);
    % psf_est = fspecial('gaussian', psfSize, sigma);

    % 2: Smarter Guess, stars and remove background
    bright_thresh = 0.75; 
    patch_size = 25;
    [psf_init, centroids, patch_stack] = estimate_star_psf(rgbSignal, patch_size, bright_thresh);

    % C: Apply PSF for (semi) blind deconvolution
    numIter = 5; NSR = 1e-2; bypass = false;
    [rgbSignal, psf_est] = smartDeconvRGB(rgbSignal, psf_init, numIter, NSR, bypass);

    % --- Display PSFs side by side ---
    debug = false;
    if debug
        figure;

        % Initial PSF
        subplot(1,2,1);
        imagesc(psf_init);
        axis image;
        colormap hot;
        colorbar;
        title('Initial PSF Guess');

        % Estimated PSF from blind deconvolution
        subplot(1,2,2);
        imagesc(psf_est);
        axis image;
        colormap hot;
        colorbar;
        title('Estimated PSF');

        sgtitle('PSF Comparison');  % Optional super title
    end

    % D: Denoise again
    %rgbSignal = denoiseRGB(rgbSignal, net);


    % ---------- SAVE ----------
    saveMode = 'fp32_scaled'; % 'fp32_raw' | 'fp32_scaled' | 'uint16_scaled'
    maxVal_img = 2^14 - 1; % Global Weight for Maximum, standard 16 bit, adjust as needed
    minVal = min(rgbSignal(:)); 
    maxVal = max(rgbSignal(:));
    fprintf('Saving: min=%.6g  max=%.6g  mode=%s\n', minVal, maxVal, saveMode);
    
    [~, baseName, ~] = fileparts(rawFiles(idx).name);
    outFile = fullfile(destDir, [baseName, '.tif']);
    
    rgbNorm = single(rgbSignal*prescaler / maxVal_img);
    
    t = Tiff(outFile, 'w');
    
    % --- create tagstruct locally inside parfor ---
    tagstruct = struct();
    tagstruct.ImageLength = size(rgbNorm,1);
    tagstruct.ImageWidth  = size(rgbNorm,2);
    tagstruct.Photometric = Tiff.Photometric.RGB;
    tagstruct.BitsPerSample = 32;
    tagstruct.SampleFormat = Tiff.SampleFormat.IEEEFP;
    tagstruct.SamplesPerPixel = 3;
    tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
    tagstruct.Compression = Tiff.Compression.None;
    tagstruct.Software = 'MATLAB';
    
    t.setTag(tagstruct);
    t.write(rgbNorm);
    t.close();
    
    fprintf('Saved FP32 (scaled to [0,%.2f]) to %s\n', max(rgbNorm(:)), outFile);

end


elapsedTime = toc;
disp(['Done! Elapsed time: ', num2str(elapsedTime), ' sec']);


%% ----------------- Local helper functions -----------------

% rawToRGB: standalone converter wrapper (reads RAW, removes hot pixels, subtracts black level, clamps, demosaics -> returns single RGB)
function rgbSignal = rawToRGB(rawFilePath, varargin)
    % Usage:
    % rgbSignal = rawToRGB(path, 'BlackLevel', [bR bG1 bG2 bB], ...
    %                     'RemoveHotPixels',true, 'ClipMax', 16383)

    p = inputParser;
    addRequired(p, 'rawFilePath', @(x) ischar(x) || isstring(x));
    addParameter(p, 'BlackLevel', [0 0 0 0], @(x) isnumeric(x) && (isscalar(x) || numel(x)==4));
    addParameter(p, 'RemoveHotPixels', true, @(x) islogical(x) || (isnumeric(x) && ismember(x,[0 1])));
    parse(p, rawFilePath, varargin{:});

    rawFilePath = char(p.Results.rawFilePath);
    blackLevel = single(p.Results.BlackLevel);
    removeHot = logical(p.Results.RemoveHotPixels);

    if isscalar(blackLevel)
        blackLevel = repmat(blackLevel,1,4);
    else
        blackLevel = reshape(blackLevel,1,4);
    end

    if ~exist(rawFilePath,'file')
        error('rawToRGB:FileNotFound', 'RAW file not found: %s', rawFilePath);
    end

    % Read CFA RAW (expects rawread on path)
    try
        cfaImage = single(rawread(rawFilePath));
    catch ME
        error('rawToRGB:rawreadFailed', ['Failed to read RAW file using rawread. ' ...
            'Ensure rawread is available and the file is a supported RAW format.\nMATLAB message: %s'], ME.message);
    end

    % Optional hot-pixel removal
    if removeHot
        cfaImage = removeHotPixelsRAW(cfaImage);
    end

    % Subtract black level (fixed RGGB pattern)
    try
        cfaImage(1:2:end,1:2:end) = cfaImage(1:2:end,1:2:end) - blackLevel(1); % R
        cfaImage(1:2:end,2:2:end) = cfaImage(1:2:end,2:2:end) - blackLevel(2); % G1
        cfaImage(2:2:end,1:2:end) = cfaImage(2:2:end,1:2:end) - blackLevel(3); % G2
        cfaImage(2:2:end,2:2:end) = cfaImage(2:2:end,2:2:end) - blackLevel(4); % B
    catch
        error('rawToRGB:IndexingError', 'Error subtracting black levels. Check CFA dimensions and provided BlackLevel.');
    end

    % Demosaic (MATLAB expects 'Rggb' with exact capitalization)
    rgbSignal = single(demosaic(uint16(cfaImage), 'Rggb'));
end

% Remove Hot Pixel function (vectorized, memory-efficient)
function cleanRawImage = removeHotPixelsRAW(cfaImage)
    % cfaImage : single-channel raw image (single or single)
    % cleanRawImage : same size & type as cfaImage, with hot pixels replaced

    cleanRawImage = cfaImage;

    % 8-neighbour averaging kernel (center = 0)
    kernel = [1 1 1; 1 0 1; 1 1 1] / 8;

    % neighbour mean
    avgNeighbors = imfilter(cfaImage, kernel, 'replicate', 'same');

    % detect hot pixels (threshold factor 1.5 like original)
    hotPixelMask = cfaImage > avgNeighbors * 1.5;

    % replace with neighbour average
    cleanRawImage(hotPixelMask) = avgNeighbors(hotPixelMask);
end

% Gaussian Low-pass Filter Function
function outputImage = applyGaussianLowPass(inputImage, minPixelDetail)
    % Return original if no filtering requested
    if minPixelDetail == 0
        outputImage = inputImage;
        return;
    end

    [rows, cols, numChannels] = size(inputImage);

    % Frequency-domain standard deviation (controls cutoff)
    sigmaFreq = max(rows, cols) / minPixelDetail;

    % Create coordinate grid
    [X, Y] = meshgrid(1:cols, 1:rows);
    centerX = ceil(cols/2);
    centerY = ceil(rows/2);

    % Gaussian low-pass mask (frequency domain)
    gaussMask = exp(-((X - centerX).^2 + (Y - centerY).^2) / (2*sigmaFreq^2));

    outputImage = zeros(rows, cols, numChannels);

    % Apply mask in Fourier domain to each channel
    for ch = 1:numChannels
        channelFFT = fft2(inputImage(:,:,ch));
        filteredChannel = ifft2(fftshift(gaussMask) .* channelFFT);
        outputImage(:,:,ch) = real(filteredChannel);
    end
end

% Function to denoise RGB with dynamic range compander
function denoisedRGB = denoiseRGB(img, net)

    % 1. Normalize to [0,1] and do sanity check
    maxVal = max(img(:));
    minVal = min(img(:));
    imgNorm = img / maxVal;

    if maxVal == 0 || minVal < 0
        error('Something is broken (nan or negative)');
    end

    % 2. Denoise each channel
    denoisedRGBNorm = zeros(size(imgNorm), 'like', imgNorm);
    for i = 1:3
        denoisedRGBNorm(:,:,i) = denoiseImage(imgNorm(:,:,i), net);
    end

    % 3. Recombine and rescale back
    denoisedRGB = denoisedRGBNorm * maxVal;
end

% Estimate PSF from Centroids with background subtraction
function [psf_est, centroids, patch_stack] = estimate_star_psf(y, patch_size, bright_thresh)
    
    % grayscale and Normalize to 0 1
    y_gray = im2gray(y); y_gray = y_gray / max(y_gray(:));

    % threshold bright pixels
    bw = y_gray > bright_thresh;
    % keep local maxima only
    bw = bw & imregionalmax(y_gray);
    
    % find centroids
    cc = bwconncomp(bw);
    props = regionprops(cc, 'Centroid');
    num_stars = numel(props);
    if num_stars == 0
        error('No stars found. Lower bright_thresh or check image.');
    end
    centroids = reshape([props.Centroid],2,[])';
    
    % prepare patch extraction
    half = floor(patch_size/2);
    y_pad = padarray(y_gray, [half half], median(y_gray(:)), 'both');
    
    % extract patches
    patch_stack = zeros(patch_size, patch_size, num_stars);
    bg = median(y_gray(:));  % global background (or local per patch)

    for k = 1:num_stars
        c = round(centroids(k,:));
        cx = c(1) + half; cy = c(2) + half;
        patch = y_pad(cy-half:cy+half, cx-half:cx+half);
        
        % subtract background & clip
        patch = patch - bg;
        patch(patch < 0) = 0;
        
        patch_stack(:,:,k) = patch;
    end

    % estimate PSF as median of stacked patches
    psf_est = median(patch_stack,3);
    psf_est = psf_est / sum(psf_est(:));  % normalize

    % Ensure PSF is non-negative
    psf_est(psf_est < 0) = 0;

end

% Blind Deconvolution Function
function [x_recovered, h_est] = smartDeconvRGB(y, h, numIter, NSR, bypass)

    if bypass
        h_est = h; % bypass
    else
        % 1: (semi) Blind deconvolution on luminance weighted to 0 1
        y_gray = im2gray(y); y_gray = y_gray / max(y_gray(:));
    
        [~, h_est] = deconvblind(y_gray, h, numIter);
    end

    % 2: Apply to other channels
    x_1 = deconvwnr(y(:,:,1), h_est, NSR);
    x_2 = deconvwnr(y(:,:,2), h_est, NSR);
    x_3 = deconvwnr(y(:,:,3), h_est, NSR);
    
    % 3: Concatenate recovered
    x_recovered = cat(3, x_1, x_2, x_3);
    
    %fix weirdness
    x_recovered(x_recovered < 0) = 0;
    x_recovered(x_recovered > 1) = 1; 
 
end

