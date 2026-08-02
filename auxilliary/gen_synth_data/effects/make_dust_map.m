function dust = make_dust_map(cfg)

    [X, Y] = meshgrid(linspace(-1, 1, cfg.global.W), linspace(-1, 1, cfg.global.H));
    dust = ones(cfg.global.H, cfg.global.W);

    for k = 1:cfg.optics.dustCount
        cx = -0.8 + 1.6 * rand();
        cy = -0.8 + 1.6 * rand();

        rad   = cfg.optics.dustRadiusFrac(1) + (cfg.optics.dustRadiusFrac(2) -cfg.optics.dustRadiusFrac(1)) * rand();
        depth = cfg.optics.dustDepthFrac(1)  + (cfg.optics.dustDepthFrac(2)  - cfg.optics.dustDepthFrac(1))  * rand();

        r2 = (X - cx).^2 + (Y - cy).^2;
        spot = depth * exp(-r2 / (2 * rad^2));

        dust = dust .* (1 - spot);
    end

    dust = max(dust, 0.05);

end