function img = add_hot_pixels(img, cfg)

    n = min(cfg.sensor.hotPixelCount, numel(img));
    if n <= 0
        return;
    end

    idx = randperm(numel(img), n);

    boost = cfg.sensor.hotPixelBoost * (1 + 0.5 * rand(1, n));
    img(idx) = img(idx) + boost;

end