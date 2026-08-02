function scene = make_scene(cfg)

    H = cfg.global.H;
    W = cfg.global.W;

    scene = zeros(H, W, 'single');

    %% --- faint background gradient ---
    [X, Y] = meshgrid(linspace(-1,1,W), linspace(-1,1,H));
    scene = scene + single(0.05 + 0.03*(X + 1)/2 + 0.02*(Y + 1)/2);

    %% --- star field ---
    nStars = round(0.002 * H * W);  % density control

    xs = randi(W, [nStars, 1]);
    ys = randi(H, [nStars, 1]);

    for i = 1:nStars
        x = xs(i);
        y = ys(i);

        % power-law-ish brightness (lots of faint, few bright)
        amp = (rand()^3) * 1.5;

        % small PSF-like spread
        sigma = 0.8 + 1.5*rand();

        % local stamp (tiny Gaussian kernel)
        rad = ceil(3*sigma);
        [xx, yy] = meshgrid(-rad:rad, -rad:rad);
        psf = exp(-(xx.^2 + yy.^2)/(2*sigma^2));

        x1 = max(1, x-rad); x2 = min(W, x+rad);
        y1 = max(1, y-rad); y2 = min(H, y+rad);

        sx1 = 1 + (x1-(x-rad));
        sy1 = 1 + (y1-(y-rad));
        sx2 = sx1 + (x2-x1);
        sy2 = sy1 + (y2-y1);

        scene(y1:y2, x1:x2) = scene(y1:y2, x1:x2) + ...
            single(amp) * psf(sy1:sy2, sx1:sx2);
    end

    %% --- faint nebulosity (low-frequency structure) ---
    lowFreq = imgaussfilt(rand(H,W,'single'), 20);
    scene = scene + 0.08 * lowFreq;

    %% --- normalize ---
    scene = scene - min(scene(:));
    scene = scene ./ max(scene(:) + eps);

end