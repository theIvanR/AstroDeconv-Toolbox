function F = getFlatChannel(masterFlat, c)
    if ndims(masterFlat) == 2
        F = masterFlat;
    elseif ndims(masterFlat) == 3
        if size(masterFlat,3) < c
            error('masterFlat has fewer channels than the light frame.');
        end
        F = masterFlat(:,:,c);
    else
        error('masterFlat must be 2D or 3D.');
    end
end
