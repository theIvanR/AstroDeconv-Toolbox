function img = apply_saturation(img, cfg)

    img = max(img, 0);
    img = min(img, cfg.sensor.saturationLevel);

end