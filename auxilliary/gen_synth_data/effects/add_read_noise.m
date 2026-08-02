function img = add_read_noise(img, cfg)

    img = img + cfg.noise.readSigma .* randn(size(img));

end