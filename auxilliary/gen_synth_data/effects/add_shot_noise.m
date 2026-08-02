function img = add_shot_noise(img, cfg)

    sigma = cfg.noise.shotScale .* sqrt(max(img, 0) + eps);
    img = img + sigma .* randn(size(img));

end