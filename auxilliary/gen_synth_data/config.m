function cfg = synthetic_config()

    scriptDir = fileparts(mfilename('fullpath'));

    %  GLOBAL / RUN SETTINGS
    cfg.global.seed = 42;
    cfg.global.H = 512;
    cfg.global.W = 512;

    %  OUTPUT
    cfg.output.root     = fullfile(scriptDir, 'output');
    cfg.output.flatDir  = fullfile(cfg.output.root, 'Flat');
    cfg.output.lightDir = fullfile(cfg.output.root, 'Light');
    cfg.output.truthDir = fullfile(cfg.output.root, 'truth');

    %  DATASET SIZE
    cfg.dataset.nFlats  = 16;
    cfg.dataset.nLights = 256;

    %  SCENE MODEL
    cfg.scene.baseLevel = 0.05;
    cfg.scene.starDensity = 0.002;
    cfg.scene.blobCount = 8;

    %  EXPOSURE MODEL
    cfg.exposure.flatBase  = 0.75;
    cfg.exposure.lightBase = 0.85;

    cfg.exposure.flatJitter  = 0.08;
    cfg.exposure.lightJitter = 0.05;

    
    %  OPTICS MODEL
    cfg.optics.vignetteStrength = 0.55;

    cfg.optics.dustCount = 6;
    cfg.optics.dustRadiusFrac = [0.010, 0.035];
    cfg.optics.dustDepthFrac  = [0.03, 0.18];

    %  OPTICS MODEL - PSF (Atmospheric Seeing)
    cfg.optics.psf.enabled     = true;       % Toggle on/off
    cfg.optics.psf.type        = 'moffat';   % Just for documentation
    
    % Baseline seeing parameters (typical values for a 1-2 arcsec seeing)
    cfg.optics.psf.alpha       = 1.2;        % HWHM in pixels (roughly FWHM/1.66 for beta~3.5)
    cfg.optics.psf.beta        = 3.5;        % Wing parameter (3.5 is classic Moffat)
    
    % Per-frame jitter (simulates changing atmospheric turbulence)
    cfg.optics.psf.alphaJitter = 0.25;       % Std dev of alpha variation (pixels)
    cfg.optics.psf.betaJitter  = 0.3;        % Std dev of beta variation
    
    % (Optional) You can still keep a fallback Gaussian legacy field if you want
    % cfg.optics.psfSigma = 0; % Deprecated, but leave it for compatibility


    % --- drift model ---
    cfg.optics.drift.enabled = true;
    
    % state (will be overwritten at runtime)
    cfg.optics.drift.x = 0;
    cfg.optics.drift.y = 0;
    cfg.optics.drift.vx = 0;
    cfg.optics.drift.vy = 0;
    
    % dynamics (AR(1) + noise)
    cfg.optics.drift.rho   = 0.90;   % memory (0.85–0.98 typical)
    cfg.optics.drift.sigma = 0.3;    % acceleration noise per frame (px)
    
    % systematic bias (polar alignment error etc.)
    cfg.optics.drift.biasX = 0.01;
    cfg.optics.drift.biasY = 0.005;
    
    % safety / realism bounds
    cfg.optics.drift.maxOffset   = 10;   % max pointing error (px)
    cfg.optics.drift.maxVelocity = 1.5;  % max drift speed (px/frame)


    %  NOISE MODEL
    cfg.noise.shotScale = 0.02;
    cfg.noise.readSigma = 0.008;

    %  SENSOR ARTIFACTS
    cfg.sensor.hotPixelCount = 25;
    cfg.sensor.hotPixelBoost = 0.75;
    cfg.sensor.saturationLevel = 1.0;

end