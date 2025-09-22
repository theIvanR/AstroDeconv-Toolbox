%% process_arw_modular.m
% Single-file pipeline:
%  - scans Input/*.arw, Flat/*.arw
%  - converts each RAW to an RGB linear image using rawToRGB()
%  - optional IR handling, AutoWB, denoise, lowpass (optional)
%  - saves TIFF outputs to Output/
%
% Requirements:
%  - rawread (or replace the call inside rawToRGB)
%  - Image Processing Toolbox (demosaic, imfilter, denoiseImage)
%  - Deep Learning Toolbox (for denoisingNetwork), or load an alternative net

clear; clc;

%% I/O
srcDir  = 'Input';  rawFiles  = dir(fullfile(srcDir, '*.arw'));
flatDir = 'Flat';   flatFiles = dir(fullfile(flatDir, '*.arw'));
destDir = 'Output'; if ~exist(destDir,'dir'), mkdir(destDir); end

%% Options
maxVal_img = 2^14 - 1;                % Global Weight Window for image normalization (adjust as needed)
blackLevel = [512 512 512 512];       % [R, G1, G2, B]

useIR = true;                         % true to halve green channel
useAutoWB = true;                     % true to perform white balance

% Hampel settings (applied on black-subtracted raw, removes hot pixels 8 average kernel) 
doHampel = true;

% LPF Settings (radius, skips if zero)
minPixelDetail = 4;                  

% CNN Denoiser (DnCNN)
net = denoisingNetwork("DnCNN");


%% Process Flat Frames 
tic
fprintf('Processing %d flat frames...\n', length(flatFiles));

if isempty(flatFiles)
    masterFlatInv = [];
else
    flatStack = [];
    for idx = 1:length(flatFiles)
        rawPath = fullfile(flatFiles(idx).folder, flatFiles(idx).name);
        fprintf('  Loading flat: %s\n', flatFiles(idx).name);

        % same pipeline as lights
        flatRGB = rawToRGB(rawPath, ...
            'BlackLevel', blackLevel, ...
            'RemoveHotPixels', true, ...
            'ClipMax', maxVal_img);

        % stack as single precision
        flatStack(:,:,:,idx) = single(flatRGB);
    end

    % combine (median is safer, mean is faster)
    masterFlat = median(flatStack, 4);

    % normalize each channel to unity
    for c = 1:3
        channelMedian = median(masterFlat(:,:,c), 'all');
        if channelMedian <= 0
            warning('Flat channel %d median <= 0, skipping normalization.', c);
            channelMedian = 1;
        end
        masterFlat(:,:,c) = masterFlat(:,:,c) / channelMedian;
    end

    % inverse flat for multiplicative correction
    tiny = 1e-12;
    masterFlatInv = 1 ./ max(masterFlat, tiny);

    fprintf('Master flat ready: size=%dx%dx3\n', size(masterFlat,1), size(masterFlat,2));
end

elapsedTime = toc;
fprintf('Flats pass done. Elapsed time: %.2f sec\n\n', elapsedTime);


%% Process Light Frames    
tic
fprintf('Processing %d Light frames...\n', length(rawFiles));

for idx = 1:length(rawFiles)
    rawFilePath = fullfile(rawFiles(idx).folder, rawFiles(idx).name);
    fprintf('\nProcessing %s (%d of %d)...\n', rawFiles(idx).name, idx, length(rawFiles));

    % Convert RAW -> linear RGB (single). rawToRGB handles hot pixel removal,
    % black subtraction and demosaic for RGGB pattern by default.
    rgbSignal = rawToRGB(rawFilePath, 'BlackLevel', blackLevel, 'RemoveHotPixels', true, 'ClipMax', maxVal_img);

    % Apply master flat correction if available
    if exist('masterFlatInv','var') && ~isempty(masterFlatInv)
        if ~isequal(size(rgbSignal), size(masterFlatInv))
            warning('Master flat size does not match light frame size — skipping flat correction.');
        else
            rgbSignal = rgbSignal .* masterFlatInv;
        end
    end

    % ----------------- AutoWB with selectable reference channel and IR -----------------
    refChannel = 'G';      % 'R', 'G', or 'B'  (reference channel whose median stays the same)
    allowAmplify = true;   % true = allow gains > 1; false = never amplify (only attenuate)

    if useIR
        % half the recorded green since IR contamination is common in some Sony sensors
        rgbSignal(:, :, 2) = rgbSignal(:, :, 2) / 2;
    end

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
        fprintf('AutoWB(ref=%s): nnz(mask)=%d raw_scales=[%.6g %.6g %.6g] gains=[%.6g %.6g %.6g] finalMax=%.6g\n', ...
            refChannel, nnz(mask), scaleR, scaleG, scaleB, gR, gG, gB, max(rgbSignal(:)));
    end


    % Apply Physics Filter (blobularity metrics from FWMH blobulity)
    %rgbSignal = physicsStreakCorrector(rgbSignal); % blob reshaper (cosmetic, fwhm fixing)
    rgbSignal = physicsDeconvolution(rgbSignal); %Proper Semi blind deconvolution

    % Apply Noise Filters
    rgbSignal = denoiseRGB(rgbSignal, net); % CNN Denoiser (very robust)

    rgbSignal = applyGaussianLowPass(rgbSignal, minPixelDetail); % LPF(skips if minPixelDetail == 0)


    % ---------- SHOW ----------
    %figure( 'Name', rawFiles(idx).name );
    %imshow(rgbSignal / (max(rgbSignal(:))), []);
    %title(sprintf('Sony ARW Linear RGGB Preview — %s', rawFiles(idx).name));

    % ---------- SAVE ----------
    saveMode = 'fp32_scaled'; % 'fp32_raw' | 'fp32_scaled' | 'uint16_scaled'
    minVal = min(rgbSignal(:));
    maxVal = max(rgbSignal(:));
    fprintf('Saving: min=%.6g  max=%.6g  mode=%s\n', minVal, maxVal, saveMode);

    [~, baseName, ~] = fileparts(rawFiles(idx).name);
    outFile = fullfile(destDir, [baseName, '.tif']);

    switch saveMode
        case 'fp32_raw'
            t = Tiff(outFile, 'w');
            tagstruct.ImageLength = size(rgbSignal,1);
            tagstruct.ImageWidth = size(rgbSignal,2);
            tagstruct.Photometric = Tiff.Photometric.RGB;
            tagstruct.BitsPerSample = 32;
            tagstruct.SampleFormat = Tiff.SampleFormat.IEEEFP;
            tagstruct.SamplesPerPixel = 3;
            tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
            tagstruct.Compression = Tiff.Compression.None;
            tagstruct.Software = 'MATLAB';
            t.setTag(tagstruct);
            t.write(single(rgbSignal));
            t.close();
            fprintf('Saved FP32 RAW to %s (may appear white in many viewers)\n', outFile);

        case 'fp32_scaled'
            if maxVal == 0
                rgbNorm = zeros(size(rgbSignal), 'single');
            else
                rgbNorm = single(rgbSignal / maxVal_img);
            end
            t = Tiff(outFile, 'w');
            tagstruct.ImageLength = size(rgbNorm,1);
            tagstruct.ImageWidth = size(rgbNorm,2);
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

        case 'uint16_scaled'
            if maxVal == 0
                rgb16 = uint16(0);
            else
                overMaxMask = rgbSignal > maxVal;
                if any(overMaxMask(:))
                    warning('Some values exceeded maxVal=%.6g and were clipped to this value.', maxVal);
                end
                rgbClipped = min(rgbSignal, maxVal);
                rgb16 = uint16(round((rgbClipped / maxVal) * 65535));
            end
            imwrite(rgb16, outFile, 'tif');
            fprintf('Saved uint16 (scaled by image max=%.6g) to %s\n', maxVal, outFile);
    end

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
    addParameter(p, 'ClipMax', 65535, @(x) isnumeric(x) && isscalar(x) && x>0);
    parse(p, rawFilePath, varargin{:});

    rawFilePath = char(p.Results.rawFilePath);
    blackLevel = single(p.Results.BlackLevel);
    removeHot = logical(p.Results.RemoveHotPixels);
    clipMax = single(p.Results.ClipMax);

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

    % Clamp
    cfaImage(cfaImage < 0) = 0;
    cfaImage(cfaImage > clipMax) = clipMax;

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
    % img : input RGB image (single, arbitrary scale)
    % net : pretrained DnCNN (or other) denoising network accepted by denoiseImage()

    % protect against all-zero image
    maxVal = max(img(:));
    if maxVal == 0
        denoisedRGB = img;
        return;
    end

    % 1. Normalize to [0,1]
    imgNorm = img / maxVal;
    imgNorm = min(max(imgNorm,0),1);

    % 2. Denoise each channel
    denoisedR = denoiseImage(imgNorm(:,:,1), net);
    denoisedG = denoiseImage(imgNorm(:,:,2), net);
    denoisedB = denoiseImage(imgNorm(:,:,3), net);

    % 3. Recombine and rescale back
    denoisedRGBNorm = cat(3, denoisedR, denoisedG, denoisedB);
    denoisedRGB = denoisedRGBNorm * maxVal;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Create Ellipse Mask for Stars (Star Reshaping to center and circle)
function rgbCorrected = physicsStreakCorrector(rgbSignal)
        % rgbSignal: linear RGB single image
        % rgbCorrected: streak-corrected output
    
        % Convert to grayscale (luminance)
        lum = 0.2989*rgbSignal(:,:,1) + 0.5870*rgbSignal(:,:,2) + 0.1140*rgbSignal(:,:,3);
    
        % Threshold bright spots (top 0.5%)
        thresh = prctile(lum(:), 99.5);
        brightMask = lum >= thresh;
    
        % Label connected components (stars)
        CC = bwconncomp(brightMask);
        stats = regionprops(CC, 'Centroid', 'MajorAxisLength', 'MinorAxisLength', 'Orientation');
    
        % Build a streak map
        streakMap = zeros(size(lum));
        for k = 1:numel(stats)
            ratio = stats(k).MinorAxisLength / stats(k).MajorAxisLength; % blobularity
            if ratio < 0.8  % adjustable threshold: low ratio = streak
                % draw an ellipse mask for this star
                mask = createEllipseMask(size(lum), stats(k).Centroid, ...
                                         stats(k).MajorAxisLength/2, ...
                                         stats(k).MinorAxisLength/2, ...
                                         stats(k).Orientation);
                streakMap(mask) = 1 - ratio; % weight by streakiness
            end
        end
    
        % Smooth streaks with anisotropic Gaussian (minor axis only)
        hsize = 5; % kernel size
        sigma = 1; % smoothing along minor axis
        streakCorrected = imgaussfilt(lum, sigma);
    
        % Blend back into RGB channels
        rgbCorrected = rgbSignal;
        for c = 1:3
            rgbCorrected(:,:,c) = rgbSignal(:,:,c) .* (1 - streakMap) + streakCorrected .* streakMap;
        end
    end
    
    % Helper: create an ellipse mask
    function mask = createEllipseMask(imSize, centroid, a, b, theta)
        [X, Y] = meshgrid(1:imSize(2), 1:imSize(1));
        x0 = centroid(1); y0 = centroid(2);
        theta = deg2rad(theta);
        Xr = (X - x0)*cos(theta) + (Y - y0)*sin(theta);
        Yr = -(X - x0)*sin(theta) + (Y - y0)*cos(theta);
        mask = (Xr.^2)/(a^2) + (Yr.^2)/(b^2) <= 1;
    end

% Proper Semi Blind Deconvolution
function rgbCorrected = physicsDeconvolution(rgbSignal)
% rgbSignal: linear RGB single image
% rgbCorrected: RGB after streak / PSF deconvolution
%
% Fully drop-in: just call rgbCorrected = physicsDeconvolution(rgbSignal);

    % ----------------- PARAMETERS -----------------
    topPercentile = 99.5;  % for bright star detection
    minBlobularity = 0.8;   % ratio minor/major axis to consider streaked
    maxStars = 20;          % max stars to consider
    psfSize = 15;           % PSF kernel size (pixels)
    rlIterations = 8;       % RL deconvolution iterations

    % ----------------- STEP 1: LUMINANCE -----------------
    lum = 0.2989*rgbSignal(:,:,1) + 0.5870*rgbSignal(:,:,2) + 0.1140*rgbSignal(:,:,3);

    % ----------------- STEP 2: BRIGHT STAR DETECTION -----------------
    thresh = prctile(lum(:), topPercentile);
    brightMask = lum >= thresh;
    CC = bwconncomp(brightMask);
    stats = regionprops(CC, 'Centroid', 'MajorAxisLength', 'MinorAxisLength', 'Orientation');

    % Sort by brightness and limit number of stars
    starValues = zeros(numel(stats),1);
    for k = 1:numel(stats)
        starValues(k) = lum(round(stats(k).Centroid(2)), round(stats(k).Centroid(1)));
    end
    [~, idxSort] = sort(starValues, 'descend');
    stats = stats(idxSort);
    if numel(stats) > maxStars
        stats = stats(1:maxStars);
    end

    % ----------------- STEP 3: BUILD AVERAGE STAR PSF -----------------
    psfAccum = zeros(psfSize, psfSize);
    count = 0;
    for k = 1:numel(stats)
        ratio = stats(k).MinorAxisLength / stats(k).MajorAxisLength;
        if ratio < minBlobularity
            % build elliptical PSF kernel
            a = stats(k).MajorAxisLength / 2;
            b = stats(k).MinorAxisLength / 2;
            theta = stats(k).Orientation;
            psfKernel = createEllipsePSF(psfSize, a, b, theta);
            psfAccum = psfAccum + psfKernel;
            count = count + 1;
        end
    end
    if count == 0
        % fallback to circular PSF
        psfKernel = fspecial('gaussian', psfSize, 1);
    else
        psfKernel = psfAccum / count;
        psfKernel = psfKernel / sum(psfKernel(:)); % normalize
    end

    % ----------------- STEP 4: DECONVOLVE EACH CHANNEL -----------------
    rgbCorrected = zeros(size(rgbSignal), 'like', rgbSignal);
    for c = 1:3
        rgbCorrected(:,:,c) = deconvlucy(rgbSignal(:,:,c), psfKernel, rlIterations);
    end
end

    % ----------------- HELPER: CREATE ELLIPSE PSF -----------------
    function psf = createEllipsePSF(psfSize, a, b, theta)
    % Generate normalized elliptical PSF kernel
    [X, Y] = meshgrid(1:psfSize, 1:psfSize);
    x0 = ceil(psfSize/2); y0 = ceil(psfSize/2);
    theta = deg2rad(theta);
    Xr = (X - x0)*cos(theta) + (Y - y0)*sin(theta);
    Yr = -(X - x0)*sin(theta) + (Y - y0)*cos(theta);
    psf = exp(-((Xr.^2)/(2*a^2) + (Yr.^2)/(2*b^2)));
    psf = psf / sum(psf(:));
end

