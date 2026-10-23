
function masterDark = make_master_dark(darkJobs, opts)
%MAKE_MASTER_DARK Construct a master dark using kappa-sigma clipping.
%
%   masterDark = make_master_dark(darkJobs)
%   masterDark = make_master_dark(darkJobs, opts)
%
%   Combines dark frames in their original intensity scale.
%   Does NOT normalize individual frames or the master dark.
%
%   Options:
%       opts.kappa   - Clipping threshold (default: 3)
%       opts.maxIter - Clipping iterations (default: 3)
%
%   Supports grayscale, RGB, and RGBA (alpha ignored).

    %% Defaults
    if nargin < 2 || isempty(opts)
        opts = struct();
    end

    if ~isfield(opts, 'kappa'),  opts.kappa = 3; end
    if ~isfield(opts, 'maxIter'), opts.maxIter = 3; end

    if isempty(darkJobs)
        error('No dark jobs provided.');
    end

    %% Load dark frames
    frames = [];
    count  = 0;
    refHW  = [];
    nCh    = [];

    for i = 1:numel(darkJobs)

        if ~darkJobs(i).exists
            continue;
        end

        file = darkJobs(i).filePath;
        img = im2single(imread(file));

        % Channel handling
        if ndims(img) == 2
            img = reshape(img, size(img,1), size(img,2), 1);

        elseif ndims(img) == 3
            if size(img,3) == 4
                warning('Dark "%s" contains alpha. Ignoring alpha.', file);
                img = img(:,:,1:3);
            end

            if ~ismember(size(img,3), [1 3])
                error('Unsupported channel count in "%s".', file);
            end
        else
            error('Unsupported image dimensions in "%s".', file);
        end

        % Geometry and channel consistency
        hw = [size(img,1), size(img,2)];
        ch = size(img,3);

        if isempty(refHW)
            refHW = hw;
            nCh = ch;
            frames = zeros(hw(1), hw(2), ch, numel(darkJobs), ...
                'single');
        else
            if ~isequal(hw, refHW)
                error('Dark size mismatch in "%s".', file);
            end

            if ch ~= nCh
                error('Channel mismatch in "%s".', file);
            end
        end

        % Preserve the original intensity scale.
        count = count + 1;
        frames(:,:,:,count) = img;
    end

    if count == 0
        error('No valid dark frames found.');
    end

    frames = frames(:,:,:,1:count);

    if count < 2
        error('Need at least 2 dark frames.');
    elseif count < 10
        warning('Low dark count (%d). Results may be noisy.', count);
    elseif count < 20
        warning('Moderate dark count (%d).', count);
    end

    %% Iterative kappa-sigma clipping
    mask = true(size(frames));

    for iter = 1:opts.maxIter

        w = sum(mask, 4);
        s1 = sum(frames .* mask, 4);
        s2 = sum((frames .* frames) .* mask, 4);

        mu = s1 ./ max(w, 1);
        denom = max(w - 1, 1);

        varx = max((s2 - w .* mu.^2) ./ denom, 0);
        sigma = sqrt(varx);

        lower = mu - opts.kappa .* sigma;
        upper = mu + opts.kappa .* sigma;

        for k = 1:count
            frame = frames(:,:,:,k);
            mask(:,:,:,k) = ...
                (frame >= lower) & (frame <= upper);
        end
    end

    %% Combine surviving pixels
    w = sum(mask, 4);
    s1 = sum(frames .* mask, 4);

    masterDark = s1 ./ max(w, 1);

    % Return grayscale as 2D
    if size(masterDark, 3) == 1
        masterDark = masterDark(:,:,1);
    end

    disp('Master dark created successfully.');
end
