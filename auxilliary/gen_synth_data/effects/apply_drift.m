function [img, cfg] = apply_drift(img, cfg)

    if ~isfield(cfg.optics, 'drift') || ~cfg.optics.drift.enabled
        return;
    end

    d = cfg.optics.drift;

    % initialize state if needed
    if ~isfield(d, 'x')
        d.x = 0; d.y = 0;
        d.vx = 0; d.vy = 0;
    end

    % stochastic velocity update (AR(1) + bias)
    d.vx = d.rho * d.vx + d.sigma * randn() + d.biasX;
    d.vy = d.rho * d.vy + d.sigma * randn() + d.biasY;

    % optional caps (recommended)
    d.vx = max(min(d.vx, d.maxVelocity), -d.maxVelocity);
    d.vy = max(min(d.vy, d.maxVelocity), -d.maxVelocity);

    % integrate position
    d.x = d.x + d.vx;
    d.y = d.y + d.vy;

    % cap position (prevents runaway drift over long sims)
    d.x = max(min(d.x, d.maxOffset), -d.maxOffset);
    d.y = max(min(d.y, d.maxOffset), -d.maxOffset);

    % apply shift (IMPORTANT: update stored state back into cfg)
    img = imtranslate(img, [d.x, d.y], 'linear', 'FillValues', 0);

    cfg.optics.drift = d;
end