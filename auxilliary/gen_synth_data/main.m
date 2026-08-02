function synthetic_main()

    scriptDir = fileparts(mfilename('fullpath'));
    addpath(genpath(scriptDir));

    cfg = synthetic_config();
    rng(cfg.global.seed, 'twister');

    % Create output folders
    if ~exist(cfg.output.root, 'dir'),     mkdir(cfg.output.root); end
    if ~exist(cfg.output.flatDir, 'dir'),  mkdir(cfg.output.flatDir); end
    if ~exist(cfg.output.lightDir, 'dir'), mkdir(cfg.output.lightDir); end
    if ~exist(cfg.output.truthDir, 'dir'), mkdir(cfg.output.truthDir); end

    % Truth fields
    scene    = make_scene(cfg);
    response = make_instrument_response(cfg);

    % Save truth
    save_u16_tiff(scene,    fullfile(cfg.output.truthDir, 'scene.tiff'));
    save_u16_tiff(response, fullfile(cfg.output.truthDir, 'response.tiff'));
    save(fullfile(cfg.output.truthDir, 'truth.mat'), 'cfg', 'scene', 'response');

    % Generate flats
    for i = 1:cfg.dataset.nFlats
        illum = cfg.exposure.flatBase * (1 + cfg.exposure.flatJitter * randn());
        img = illum * response;

        img = add_shot_noise(img, cfg);
        img = add_read_noise(img, cfg);
        img = add_hot_pixels(img, cfg);
        img = apply_saturation(img, cfg);

        save_u16_tiff(img, fullfile(cfg.output.flatDir, sprintf('flat_%03d.tiff', i)));
    end

    % Generate lights (physics, blur, noise, NOT COMMUTATIVE)
    for i = 1:cfg.dataset.nLights
        exposure = cfg.exposure.lightBase * (1 + cfg.exposure.lightJitter * randn());
    
        img = exposure * scene;
    
        % Optics
        img = apply_psf(img, cfg);
        [img, cfg] = apply_drift(img, cfg);
    
        % Instrument response
        img = img .* response;
    
        % Noise / artifacts
        img = add_shot_noise(img, cfg);
        img = add_read_noise(img, cfg);
        img = add_hot_pixels(img, cfg);
        img = apply_saturation(img, cfg);
    
        save_u16_tiff(img, fullfile(cfg.output.lightDir, sprintf('light_%03d.tiff', i)));
    end

    disp("Synthetic dataset written to: " + cfg.output.root);

end