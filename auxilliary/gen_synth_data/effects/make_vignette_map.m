function vignette = make_vignette_map(cfg)

    [X, Y] = meshgrid(linspace(-1, 1, cfg.global.W), linspace(-1, 1, cfg.global.H));
    r = sqrt(X.^2 + Y.^2);

    vignette = 1 - cfg.optics.vignetteStrength * (r.^2);

    % Keep it positive
    vignette = max(vignette, 0.15);

end