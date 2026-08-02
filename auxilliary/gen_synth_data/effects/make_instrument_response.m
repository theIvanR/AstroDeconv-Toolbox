function response = make_instrument_response(cfg)

    vignette = make_vignette_map(cfg);
    dust     = make_dust_map(cfg);

    response = vignette .* dust;

    % Normalize mean response to 1
    response = response ./ mean(response(:));

end