%MAKE_MASTER_FLAT Construct a master flat frame using kappa-sigma clipping.
%
%   masterFlat = MAKE_MASTER_FLAT(flatJobs)
%   masterFlat = MAKE_MASTER_FLAT(flatJobs, opts)
%
%   Creates a master flat field from a collection of flat frames. Each
%   input frame is normalized by its mean intensity before stacking to
%   remove exposure and illumination variations while preserving relative
%   channel response.
%
%   The function supports both monochrome and RGB flat frames:
%
%       - Grayscale inputs produce a 2D master flat.
%       - RGB inputs produce a 3-channel master flat.
%       - RGBA inputs generate a warning and the alpha channel is ignored.
%       - Any other channel count results in an error.
%
%   Flat frames must all have identical dimensions and channel counts.
%
%   Processing Pipeline
%   -------------------
%   1. Load and validate flat frames.
%   2. Normalize each frame by its global mean intensity.
%   3. Perform iterative kappa-sigma clipping across the frame stack.
%   4. Compute a robust average of surviving pixels.
%   5. Normalize the final master flat to unit mean.

function masterFlat = make_master_flat(flatJobs, opts)

    %% Defaults
    if nargin < 2 || isempty(opts)
        opts = struct();
    end

    if ~isfield(opts, 'kappa'),   opts.kappa = 3; end
    if ~isfield(opts, 'maxIter'),  opts.maxIter = 3; end

    if isempty(flatJobs)
        error('No flat jobs provided.');
    end

    %% Load flats
    frames = [];
    count  = 0;
    refHW  = [];
    nCh    = [];

    for i = 1:numel(flatJobs)

        if ~flatJobs(i).exists
            continue;
        end

        file = flatJobs(i).filePath;

        % Read as floating point, then store as single for speed/memory
        img = single(im2double(imread(file)));

        % Channel handling
        if ndims(img) == 2
            img = reshape(img, size(img,1), size(img,2), 1);

        elseif ndims(img) == 3
            if size(img,3) == 4
                warning('Flat "%s" contains alpha channel. Ignoring alpha.', file);
                img = img(:,:,1:3);
            end

            if ~ismember(size(img,3), [1 3])
                error('Unsupported channel count in "%s". Expected 1, 3, or 4 channels.', file);
            end

        else
            error('Unsupported image dimensions in "%s".', file);
        end

        % Geometry + channel consistency
        hw = [size(img,1), size(img,2)];
        ch = size(img,3);

        if isempty(refHW)
            refHW = hw;
            nCh   = ch;
            frames = zeros(refHW(1), refHW(2), nCh, numel(flatJobs), 'single');
        else
            if ~isequal(hw, refHW)
                error('Flat size mismatch in "%s". Expected [%d %d], got [%d %d].', ...
                    file, refHW(1), refHW(2), hw(1), hw(2));
            end
            if ch ~= nCh
                error('Channel mismatch in "%s". Expected %d channels, got %d.', ...
                    file, nCh, ch);
            end
        end

        % Normalize each flat by one scalar
        m = mean(img, 'all');

        if ~(m > 0)
            error('Flat "%s" has non-positive mean intensity.', file);
        end

        img = img ./ m;

        count = count + 1;
        frames(:,:,:,count) = img;

    end

    if count == 0
        error('No valid flat frames found.');
    end

    frames = frames(:,:,:,1:count);

    if count < 2
        error('Need at least 2 flat frames.');
    elseif count < 10
        warning('Low flat count (%d). Results may be noisy.', count);
    elseif count < 20
        warning('Moderate flat count (%d). OK but not optimal.', count);
    end

    %% Kappa-sigma clipping
    mask = true(size(frames));

    for iter = 1:opts.maxIter

        % Weighted stats without NaNs
        w = sum(mask, 4);                          % H x W x C
        s1 = sum(frames .* mask, 4);               % sum(x)
        s2 = sum((frames .* frames) .* mask, 4);    % sum(x^2)

        mu = s1 ./ max(w, 1);

        % Sample variance from masked sums
        denom = max(w - 1, 1);
        varx  = max((s2 - w .* (mu .* mu)) ./ denom, 0);
        sigma = sqrt(varx);

        lower = mu - opts.kappa .* sigma;
        upper = mu + opts.kappa .* sigma;

        % Update per-frame mask
        for k = 1:count
            frame = frames(:,:,:,k);
            mask(:,:,:,k) = (frame >= lower) & (frame <= upper);
        end

    end

    %% Final robust combine
    w = sum(mask, 4);
    s1 = sum(frames .* mask, 4);

    masterFlat = s1 ./ max(w, 1);

    %% Final normalization
    masterMean = mean(masterFlat, 'all');

    if ~(masterMean > 0)
        error('Master flat has non-positive mean intensity.');
    end

    masterFlat = masterFlat ./ masterMean;

    %% Return grayscale as 2D
    if size(masterFlat, 3) == 1
        masterFlat = masterFlat(:,:,1);
    end

end