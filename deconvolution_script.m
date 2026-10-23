
%% MAIN — Empirical PSF debugging pipeline
clearvars;
close all;
clc;

%% Paths
rootDir = pwd;
addpath(genpath(fullfile(rootDir, 'functions')));

imageDir = fullfile(rootDir, 'output');
imagePath = fullfile(imageDir, 'autosave.tif');

%% 0: Configuration
cfg.minStarValue = 0.60;
cfg.maxStarValue = 0.85;
cfg.patchRadius = 7;
cfg.minStarDistance = 2 * cfg.patchRadius + 1;
cfg.maxStars = 100;
cfg.saturationThreshold = 0.999;

cfg.iterations = 4; % for deconvolution

cfg.channel = 0;       % 0 = all channels; 1 = R, 2 = G, 3 = B


%% Load source image
[img, count] = load_image(imagePath);

img = neutralize_background(img, 0.9); % neutralize minus 0.9*median subtraction (skyglow)

% Select channels to process.
if cfg.channel == 0
    channels = 1:count;
else
    if ~isscalar(cfg.channel) || ...
            ~isfinite(cfg.channel) || ...
            cfg.channel ~= fix(cfg.channel) || ...
            cfg.channel < 1 || cfg.channel > count
        error('cfg.channel must be 0 or an integer from 1 to %d.', count);
    end

    channels = cfg.channel;
end

% Allocate output and preserve untouched channels.
restored = img;

%% Process selected channels
for c = channels
    fprintf('\n===== Processing channel %d / %d =====\n', c, count);

    % Extract current channel.
    channel = img(:, :, c);

    % Detect stars.
    stars = detect_star_candidates(channel, cfg);

    % Extract star patches.
    patchStack = extract_star_patches(channel, stars, cfg);

    % Build empirical PSF.
    averagePSF = build_average_psf(patchStack);

    % Display PSF diagnostics.
    display_psf_debug(channel, stars, patchStack, averagePSF);

    fprintf('Detected stars: %d\n', size(stars.centers, 1));
    fprintf('Accepted patches: %d\n', size(patchStack, 3));
    fprintf('PSF dimensions: %d x %d\n', size(averagePSF, 1), size(averagePSF, 2));
    fprintf('PSF sum: %.8f\n', sum(averagePSF(:)));

    % Deconvolve.
    deconv = deconvolve_channel(channel, averagePSF, cfg.iterations);
    restored(:, :, c) = deconv;

    % Compare at identical display limits.
    figure('Name', sprintf('Channel %d Deconvolution', c),'Color', 'w');

    displayRange = [min(channel(:)), max(channel(:))];

    subplot(1, 2, 1);
    imshow(channel, displayRange);
    title(sprintf('Input channel %d', c));

    subplot(1, 2, 2);
    imshow(deconv, displayRange);
    title(sprintf('Richardson-Lucy (%d iterations)', cfg.iterations));

    fprintf('Input range:  %.6f to %.6f\n', min(channel(:)), max(channel(:)));
    fprintf('Output range: %.6f to %.6f\n', min(deconv(:)), max(deconv(:)));
end

%% Save assembled output once
outputPath = fullfile(imageDir, 'deconvolved.tif');

% Preserve the original channel layout and nominal [0,1] scale.
outputImage = uint16(min(max(restored, 0), 1) * 65535);
imwrite(outputImage, outputPath);

fprintf('\nProcessing complete.\n');
fprintf('Processed channels: %s\n', mat2str(channels));
fprintf('Saved image: %s\n', outputPath);


%% Helpers neutralization

function centered = neutralize_background(img, factor)
%NEUTRALIZE_BACKGROUND Subtract a fraction of each channel's median.
%
% Inputs:
%   img    - HxW grayscale or HxWxC image, preferably linear.
%   factor - Background subtraction factor (default 0.9).
%
% Output:
%   centered - Background-corrected image.
%
% Note: Negative values are preserved.

if nargin < 2
    factor = 0.9;
end

if ~isscalar(factor) || ~isfinite(factor) || ...
        factor < 0 || factor > 1
    error('factor must be a finite scalar between 0 and 1.');
end

centered = single(img);

for c = 1:size(centered, 3)
    channel = centered(:, :, c);
    background = median(channel(:));

    %centered(:, :, c) = channel - factor * background;
    centered(:, :, c) = max(channel - factor * background, 0);

    fprintf('Channel %d: median background = %.6f\n', c, background);
end
end

%% Helpers PSF

function [img, count] = load_image(imagePath)
%LOAD_IMAGE Load an image as double-precision intensity data.

assert(isfile(imagePath), 'Input image not found: %s', imagePath);

img = im2double(imread(imagePath));
count = size(img, 3);

if any(~isfinite(img(:)))
    error('Image contains NaN or Inf values.');
end

fprintf('Loaded image: %d x %d x %d\n', size(img, 1), size(img, 2), count);
fprintf('Intensity range: %.5f to %.5f\n', min(img(:)), max(img(:)));

end

function stars = detect_star_candidates(channel, cfg)
%DETECT_STAR_CANDIDATES Detect and select isolated unsaturated stars.
%
% Inputs:
%   green - Normalized green-channel image.
%   cfg   - Configuration struct.
%
% Output:
%   stars.centers   - Nx2 subpixel [x,y] coordinates.
%   stars.peakValues - Peak intensities.
%   stars.patchRadius - Patch radius used for edge checks.

    mask = channel >= cfg.minStarValue;

    CC = bwconncomp(mask, 8);
    stats = regionprops(CC, channel, 'Area', 'Centroid', 'MaxIntensity');

    valid = false(numel(stats), 1);

    for k = 1:numel(stats)
        valid(k) = stats(k).MaxIntensity >= cfg.minStarValue && stats(k).MaxIntensity <= cfg.maxStarValue;
    end

    stats = stats(valid);

    fprintf('Candidate objects: %d\n', numel(stats));

    if isempty(stats)
        error(['No stars found in the specified intensity range. ', 'Check cfg.minStarValue and cfg.maxStarValue.']);
    end

    [H, W] = size(channel);
    r = cfg.patchRadius;

    centers = zeros(0, 2);
    peakValues = zeros(0, 1);

    for k = 1:numel(stats)
        cx = stats(k).Centroid(1);
        cy = stats(k).Centroid(2);

        ix = round(cx);
        iy = round(cy);

        if ix-r < 1 || ix+r > W || iy-r < 1 || iy+r > H
            continue;
        end

        patch = channel(iy-r:iy+r, ix-r:ix+r);

        if max(patch(:)) >= cfg.saturationThreshold
            continue;
        end

        perimeter = false(size(patch));
        perimeter([1 end], :) = true;
        perimeter(:, [1 end]) = true;

        background = median(patch(perimeter));
        signal = max(patch - background, 0);

        [X, Y] = meshgrid(-r:r, -r:r);
        central = X.^2 + Y.^2 <= (r/2)^2;
        weights = signal .* central;

        totalWeight = sum(weights(:));

        if totalWeight <= eps
            continue;
        end

        dx = sum(X(:) .* weights(:)) / totalWeight;
        dy = sum(Y(:) .* weights(:)) / totalWeight;

        centers(end+1, :) = [ix + dx, iy + dy]; 
        peakValues(end+1, 1) = stats(k).MaxIntensity; 
    end

    if isempty(centers)
        error('No usable stars remain after edge and saturation checks.');
    end

    % Prioritize brighter stars.
    [~, order] = sort(peakValues, 'descend');
    centers = centers(order, :);
    peakValues = peakValues(order);

    % Reject stars too close to previously selected stars.
    selectedCenters = zeros(0, 2);
    selectedPeaks = zeros(0, 1);
    minDistance = cfg.minStarDistance;

    for k = 1:size(centers, 1)
        c = centers(k, :);

        if isempty(selectedCenters)
            keep = true;
        else
            d = hypot(selectedCenters(:,1) - c(1), ...
                      selectedCenters(:,2) - c(2));
            keep = all(d >= minDistance);
        end

        if keep
            selectedCenters(end+1, :) = c; 
            selectedPeaks(end+1, 1) = peakValues(k); 
        end

        if size(selectedCenters, 1) >= cfg.maxStars
            break;
        end
    end

    stars.centers = selectedCenters;
    stars.peakValues = selectedPeaks;
    stars.patchRadius = r;

    fprintf('Selected isolated stars: %d\n', ...
        size(stars.centers, 1));

    if isempty(stars.centers)
        error('No isolated stars found. Reduce cfg.minStarDistance.');
    end
end

function patchStack = extract_star_patches(channel, stars, cfg)
%EXTRACT_STAR_PATCHES Extract and normalize aligned star patches.
%
% Inputs:
%   green - Normalized green-channel image.
%   stars - Output from detect_star_candidates.
%   cfg   - Configuration struct.
%
% Output:
%   patchStack - HxWxN array of normalized star patches.

r = cfg.patchRadius;
patchSize = 2*r + 1;
nStars = size(stars.centers, 1);

[H, W] = size(channel);

patchStack = zeros(patchSize, patchSize, nStars);
valid = false(nStars, 1);

[XqOffset, YqOffset] = meshgrid(-r:r, -r:r);

for k = 1:nStars
    cx = stars.centers(k, 1);
    cy = stars.centers(k, 2);

    Xq = cx + XqOffset;
    Yq = cy + YqOffset;

    if min(Xq(:)) < 1 || max(Xq(:)) > W || ...
            min(Yq(:)) < 1 || max(Yq(:)) > H
        continue;
    end

    patch = interp2(channel, Xq, Yq, 'linear');

    if any(~isfinite(patch(:)))
        continue;
    end

    % Reject clipped patches.
    if max(patch(:)) >= cfg.saturationThreshold
        continue;
    end

    perimeter = false(size(patch));
    perimeter([1 end], :) = true;
    perimeter(:, [1 end]) = true;

    background = median(patch(perimeter));
    patch = max(patch - background, 0);

    totalFlux = sum(patch(:));

    if totalFlux <= eps
        continue;
    end

    patchStack(:, :, k) = patch / totalFlux;
    valid(k) = true;
end

patchStack = patchStack(:, :, valid);

fprintf('Valid normalized patches: %d / %d\n', ...
    size(patchStack, 3), nStars);

if isempty(patchStack)
    error('No valid star patches could be extracted.');
end
end

function averagePSF = build_average_psf(patchStack)
%BUILD_AVERAGE_PSF Construct a normalized empirical PSF.
%
% Input:
%   patchStack - HxWxN array of normalized star patches.
%
% Output:
%   averagePSF - Normalized mean PSF.

if isempty(patchStack) || size(patchStack, 3) == 0
    error('patchStack contains no valid patches.');
end

averagePSF = mean(patchStack, 3);

totalFlux = sum(averagePSF(:));

if ~isfinite(totalFlux) || totalFlux <= eps
    error('Invalid average PSF normalization.');
end

averagePSF = averagePSF / totalFlux;

fprintf('PSF dimensions: %d x %d\n', ...
    size(averagePSF, 1), size(averagePSF, 2));
fprintf('PSF flux sum: %.10f\n', sum(averagePSF(:)));
end

function display_psf_debug(green, stars, patchStack, averagePSF)
%DISPLAY_PSF_DEBUG Visualize star detection and empirical PSF.

figure('Name', 'Empirical Star PSF', 'Color', 'w');

subplot(1, 3, 1);
imshow(green, []);
title('Green channel');
hold on;

if ~isempty(stars.centers)
    plot(stars.centers(:,1), stars.centers(:,2), ...
        'r+', 'MarkerSize', 8, 'LineWidth', 1);
end

hold off;

subplot(1, 3, 2);
imshow(averagePSF, []);
title(sprintf('Average PSF (%d stars)', size(patchStack, 3)));

subplot(1, 3, 3);
imagesc(log10(max(averagePSF, 1e-8)));
axis image;
colorbar;
title('PSF log scale');
colormap(gca, parula);

figure('Name', 'Average PSF Surface', 'Color', 'w');
surf(averagePSF, 'EdgeColor', 'none');
axis tight;
view(45, 35);
colorbar;
title('Empirical PSF surface');
xlabel('X (pixels)');
ylabel('Y (pixels)');
zlabel('Normalized intensity');
end

%% Helpers Deconvolve
function restored = deconvolve_channel(image, psf, iterations)
%DECONVOLVE_CHANNEL Richardson-Lucy deconvolution using an empirical PSF.
%
% Inputs:
%   image      - Input channel, preferably linear intensity data.
%   psf        - Normalized empirical PSF.
%   iterations - Number of Richardson-Lucy iterations.
%
% Output:
%   restored   - Deconvolved image.

if nargin < 3
    iterations = 10;
end

image = double(image);
psf = double(psf);

if any(~isfinite(image(:))) || any(~isfinite(psf(:)))
    error('Image and PSF must contain only finite values.');
end

if any(image(:) < 0)
    error('Richardson-Lucy requires nonnegative image data.');
end

if any(psf(:) < 0) || sum(psf(:)) <= 0
    error('PSF must be nonnegative with positive total flux.');
end

psf = psf / sum(psf(:));

% Richardson-Lucy iterative restoration.
restored = deconvlucy(image, psf, iterations);
end
