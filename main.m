%% process_arw_modular.m
% Single-file pipeline:
%  - scans Input/*.arw, Flat/*.arw
%  - converts each RAW to an RGB linear image using rawToRGB()
%  - optional IR handling, AutoWB, denoise, lowpass (optional)
%  - saves TIFF outputs to Output/

clear; clc; close all; 

%% I/O
srcDir  = 'Input';  rawFiles  = dir(fullfile(srcDir, '*.arw'));
flatDir = 'Flat';   flatFiles = dir(fullfile(flatDir, '*.arw'));
destDir = 'Output'; if ~exist(destDir,'dir'), mkdir(destDir); end

%% Options
blackLevel = [512 512 512 512];       % [R, G1, G2, B]
useAutoWB = true;                     % true to perform white balance

% Hampel settings (applied on black-subtracted raw, removes hot pixels 8 average kernel) 
doHampel = true;              

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
        flatRGB = rawToRGB(rawPath, 'BlackLevel', blackLevel, 'RemoveHotPixels', true);

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
    rgbSignal = rawToRGB(rawFilePath, 'BlackLevel', blackLevel, 'RemoveHotPixels', true);

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

    % Normalize to [0, 1] after resize
    fprintf('Normalized to [0, 1]\n')
    prescaler = max(rgbSignal(:)); 
    rgbSignal = rgbSignal()/prescaler;

    % ----------------- Apply Denoising and Image Processing Methids Here -----------------

    % A: Lowpass and resize
    rgbSignal = applyGaussianLowPass(rgbSignal, 3); %minPixelDetail = 3 pixel
    rgbSignal = imresize(rgbSignal, 0.5, 'lanczos3');
    
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
    numIter = 50; NSR = 1e-2;
    [rgbSignal, psf_est] = blindDeconvRGB(rgbSignal, psf_init, numIter, NSR);

    % --- Display PSFs side by side ---
        % figure;
        % 
        % % Initial PSF
        % subplot(1,2,1);
        % imagesc(psf_init);
        % axis image;
        % colormap hot;
        % colorbar;
        % title('Initial PSF Guess');
        % 
        % % Estimated PSF from blind deconvolution
        % subplot(1,2,2);
        % imagesc(psf_est);
        % axis image;
        % colormap hot;
        % colorbar;
        % title('Estimated PSF');
        % 
        % sgtitle('PSF Comparison');  % Optional super title


    % D: Denoise again
    %rgbSignal = denoiseRGB(rgbSignal, net);


    % ---------- SAVE ----------
    saveMode = 'fp32_scaled'; % 'fp32_raw' | 'fp32_scaled' | 'uint16_scaled'
    maxVal_img = 2^14 - 1; % Global Weight Window for image normalization (adjust as needed)
    minVal = min(rgbSignal(:)); maxVal = max(rgbSignal(:));
    fprintf('Saving: min=%.6g  max=%.6g  mode=%s\n', minVal, maxVal, saveMode);

    [~, baseName, ~] = fileparts(rawFiles(idx).name);
    outFile = fullfile(destDir, [baseName, '.tif']);

    %fix up this mess!
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
                rgbNorm = single(rgbSignal*prescaler / maxVal_img);
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
        error('Somethins broken');
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
function [x_recovered, h_est] = blindDeconvRGB(y, h, numIter, NSR)

    % 1: (semi) Blind deconvolution on luminance weighted to 0 1
    y_gray = im2gray(y); y_gray = y_gray / max(y_gray(:));

    [~, h_est] = deconvblind(y_gray, h, numIter);
    %h_est = h;

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

