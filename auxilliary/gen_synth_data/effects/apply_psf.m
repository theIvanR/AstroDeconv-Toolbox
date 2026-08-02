% function img = apply_psf(img, cfg)
% 
%     % Gaussian atmospheric seeing PSF
%     sigma = cfg.optics.psfSigma;
% 
%     if sigma <= 0
%         return;
%     end
% 
%     k = fspecial('gaussian', [9 9], sigma);
% 
%     img = imfilter(img, k, 'replicate');
% 
% end

function img = apply_psf(img, cfg)

    % Check if PSF is enabled
    if ~isfield(cfg.optics, 'psf') || ~cfg.optics.psf.enabled
        return;
    end

    p = cfg.optics.psf;

    % --- Draw random parameters for THIS frame (Seeing variations) ---
    % Alpha (HWHM) - additive jitter, clamped to physical bounds
    alpha = p.alpha + p.alphaJitter * randn();
    alpha = max(alpha, 0.3);  % Never let it collapse to zero

    % Beta (wing steepness) - additive jitter
    beta = p.beta + p.betaJitter * randn();
    beta = max(beta, 2.0);    % Beta < 2 gives infinite integral (unphysical)

    % --- Build the Moffat kernel ---
    % Kernel size: automatically scale to capture the PSF wings
    % Moffat drops off as r^(-2*beta). For beta~3, 6*alpha captures 99% flux.
    kSize = 2 * ceil(6 * alpha) + 1;
    if mod(kSize, 2) == 0
        kSize = kSize + 1; % Ensure odd
    end
    % Clamp minimum size
    kSize = max(kSize, 5);

    % Create coordinate grid
    half = floor(kSize / 2);
    [x, y] = meshgrid(-half:half, -half:half);
    r = sqrt(x.^2 + y.^2);

    % Moffat formula: I(r) = (beta-1)/(pi*alpha^2) * [1 + (r/alpha)^2]^(-beta)
    % Normalization ensures sum(k) == 1 for any alpha/beta
    norm = (beta - 1) / (pi * alpha^2);
    k = norm * (1 + (r / alpha).^2).^(-beta);

    % Safety check: if alpha is tiny, the central pixel might dominate; fine.
    k = k / sum(k(:)); % Ensure exact unity

    % --- Apply convolution ---
    img = imfilter(img, k, 'replicate', 'conv');

end