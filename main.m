clear; clc; 

%% Options
maxVal_img = 2^14 - 1;
blackLevel = [512 512 512 512]; % [R, G1, G2, B]

useIR = false;        % true to halve green channel
useAutoWB = true;     % true to perform white balance

minPixelDetail = 10;   % LPF radius in pixels

srcDir = 'Input';
destDir = 'Output';

rawFiles = dir(fullfile(srcDir, '*.arw'));
disp(['Found ', num2str(length(rawFiles)), ' RAW files.']);

tic
for idx = 1:length(rawFiles)
    % 0: Prepare Raw files (Hampel Outlier Filter)
    rawFilePath = fullfile(rawFiles(idx).folder, rawFiles(idx).name);
    fprintf('\nProcessing %s (%d of %d)...\n', rawFiles(idx).name, idx, length(rawFiles));

    % Load RAW CFA and Apply Filter
    cfaImage = single(rawread(rawFilePath));
    cfaImage = removeHotPixelsRAW(cfaImage);
    
    % Offset from camera adc correction
    cfaImage(1:2:end,1:2:end) = cfaImage(1:2:end,1:2:end) - blackLevel(1); % R
    cfaImage(1:2:end,2:2:end) = cfaImage(1:2:end,2:2:end) - blackLevel(2); % G1
    cfaImage(2:2:end,1:2:end) = cfaImage(2:2:end,1:2:end) - blackLevel(3); % G2
    cfaImage(2:2:end,2:2:end) = cfaImage(2:2:end,2:2:end) - blackLevel(4); % B
    cfaImage(cfaImage < 0) = 0;


    % 1: Demosaic RGGB and apply options
    rgbSignal = double(demosaic(uint16(cfaImage), 'Rggb'));     % IMPORTANT: convert to double immediately for arithmetic
    clear cfaImage

    % ----------------- IR -----------------
    if useIR
        rgbSignal(:, :, 2) = rgbSignal(:, :, 2) / 2;
    end

    % ----------------- AutoWB with selectable reference channel -----------------
    refChannel = 'G';      % 'R', 'G', or 'B'  (reference channel whose median stays the same)
    allowAmplify = true;   % true = allow gains > 1; false = never amplify (only attenuate)
    
    if useAutoWB
        % build bright-pixel mask (robust)
        grayLevels = sum(rgbSignal, 3);
        threshold = prctile(double(grayLevels(:)), 99.5);
        mask = grayLevels >= threshold;
        if ~any(mask(:))
            threshold = prctile(double(grayLevels(:)), 99.0);
            mask = grayLevels >= threshold;
        end
        if ~any(mask(:))
            mask = true(size(grayLevels));
        end
    
        % extract channels
        R = rgbSignal(:,:,1); G = rgbSignal(:,:,2); B = rgbSignal(:,:,3);
    
        % compute medians on the mask
        scaleR = median(double(R(mask)));
        scaleG = median(double(G(mask)));
        scaleB = median(double(B(mask)));
    
        % defend against degenerate medians
        tiny = 1e-12;
        if ~isfinite(scaleR) || scaleR <= tiny, scaleR = median(double(R(:))) + tiny; end
        if ~isfinite(scaleG) || scaleG <= tiny, scaleG = median(double(G(:))) + tiny; end
        if ~isfinite(scaleB) || scaleB <= tiny, scaleB = median(double(B(:))) + tiny; end
    
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
        fprintf('AutoWB(ref=%s): nnz(mask)=%d raw_scales=[%.4g %.4g %.4g] gains=[%.4g %.4g %.4g] finalMax=%.6g\n', ...
            refChannel, nnz(mask), scaleR, scaleG, scaleB, gR, gG, gB, max(rgbSignal(:)));
    end


    % 2: Apply Additional Filters (Lowpass/Denoise, etc)
    %rgbSignal = applyGaussianLowPass(rgbSignal, minPixelDetail);
    
    %net = denoisingNetwork("DnCNN");
    %rgbSignal = denoiseRGB(rgbSignal, net);


    % 3: Display (linear), scaled
    figure( 'Name', rawFiles(idx).name );
    imshow(rgbSignal / (max(rgbSignal(:))), []);
    title(sprintf('Sony ARW Linear RGGB Preview — %s', rawFiles(idx).name));


    % 4: Save processed image as FP32 TIFF using Tiff class
    
    % ---------- Save options ----------
    % 'fp32_raw'     : write FP32 literally (no scaling) - many viewers show white
    % 'fp32_scaled'  : scale to [0,1] then save FP32 (good for viewing, linear preserved)
    % 'uint16_scaled': scale to [0,1] then save uint16 (widely compatible)
    saveMode = 'fp32_scaled';
    % ----------------------------------
    
    % diagnostics
    minVal = min(rgbSignal(:));
    maxVal = max(rgbSignal(:));
    fprintf('Saving: min=%.6g  max=%.6g  mode=%s\n', minVal, maxVal, saveMode);
    
    [~, baseName, ~] = fileparts(rawFiles(idx).name);
    outFile = fullfile(destDir, [baseName, '.tif']);
    
    switch saveMode
        case 'fp32_raw'
            % Write FP32 raw (no scaling). Viewer may show white unless viewer expects raw FP32.
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
            % Normalize to [0,1] (preserve relative linear intensities), then save FP32
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
            fprintf('Saved FP32 (scaled to [0,1]) to %s\n', outFile);
    
        case 'uint16_scaled'
            % Normalize to [0,1] then scale to 16-bit (0..65535). Good viewer compatibility.
            if maxVal == 0
                rgb16 = uint16(0);
            else
                % Clip to [0, maxVal] and detect if clipping occurs
                overMaxMask = rgbSignal > maxVal;
                if any(overMaxMask(:))
                    warning('Some values exceeded maxVal=%.6g and were clipped to this value.', maxVal);
                end
                rgbClipped = min(rgbSignal, maxVal);  % clip huge positives
                rgb16 = uint16(round((rgbClipped / maxVal) * 65535));
            end
            imwrite(rgb16, outFile, 'tif');
            fprintf('Saved uint16 (scaled by image max=%.6g) to %s\n', maxVal, outFile);
    end


end

elapsedTime = toc;
disp(['Done! Elapsed time: ', num2str(elapsedTime), ' sec']);



%% Helpers

% Function to denoise RGB with dynamic range compander
function denoisedRGB = denoiseRGB(img, net)
    % img : input RGB image, any double range
    % net : pretrained DnCNN network
    
    % 1. Compute original max per channel
    maxVal = max(img(:));  % overall max for expansion
    
    % 2. Compress to [0,1] (clip just in case)
    imgNorm = img / maxVal;
    imgNorm = min(max(imgNorm,0),1);
    
    % 3. Apply denoising per channel
    denoisedR = denoiseImage(imgNorm(:,:,1), net);
    denoisedG = denoiseImage(imgNorm(:,:,2), net);
    denoisedB = denoiseImage(imgNorm(:,:,3), net);
    
    % 4. Recombine channels
    denoisedRGBNorm = cat(3, denoisedR, denoisedG, denoisedB);
    
    % 5. Expand back to original scale
    denoisedRGB = denoisedRGBNorm * maxVal;
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

    % Gaussian low-pass mask
    gaussMask = exp(-((X - centerX).^2 + (Y - centerY).^2) / (2*sigmaFreq^2));

    % Initialize output
    outputImage = zeros(rows, cols, numChannels);

    % Apply Gaussian mask in Fourier domain to each channel
    for ch = 1:numChannels
        channelFFT = fft2(inputImage(:,:,ch));
        filteredChannel = ifft2(fftshift(gaussMask) .* channelFFT);
        outputImage(:,:,ch) = real(filteredChannel);  % retain real part
    end
end


% Remove Hot Pixel function
function cleanRawImage = removeHotPixelsRAW(cfaImage)
    [M, N] = size(cfaImage);  % Get the dimensions of the RAW image (single-channel)
    
    % Initialize the cleaned RAW image
    cleanRawImage = cfaImage;
    
    % Create shifted versions of the image to get the neighbors
    neighborsUp    = [cfaImage(1,:); cfaImage(1:M-1,:)];      % Shift image down
    neighborsDown  = [cfaImage(2:M,:); cfaImage(M,:)];        % Shift image up
    neighborsLeft  = [cfaImage(:,1), cfaImage(:,1:N-1)];      % Shift image right
    neighborsRight = [cfaImage(:,2:N), cfaImage(:,N)];        % Shift image left
    neighborsUL    = [neighborsUp(:,1), neighborsUp(:,1:N-1)];  % Upper left diagonal
    neighborsUR    = [neighborsUp(:,2:N), neighborsUp(:,N)];    % Upper right diagonal
    neighborsDL    = [neighborsDown(:,1), neighborsDown(:,1:N-1)]; % Lower left diagonal
    neighborsDR    = [neighborsDown(:,2:N), neighborsDown(:,N)];   % Lower right diagonal

    % Average the neighbors for each pixel
    avgNeighbors = (neighborsUp + neighborsDown + neighborsLeft + neighborsRight + ...
                    neighborsUL + neighborsUR + neighborsDL + neighborsDR) / 8;

    % Detect hot pixels by comparing to average neighbors
    hotPixelMask = cfaImage > avgNeighbors * 1.5;

    % Replace hot pixels with the average of their neighbors
    cleanRawImage(hotPixelMask) = avgNeighbors(hotPixelMask);
end
