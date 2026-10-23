
function D = getDarkChannel(masterDark, c)
%GETDARKCHANNEL Extract a channel from a master dark frame.

if ndims(masterDark) == 2
    D = masterDark;

elseif ndims(masterDark) == 3
    if size(masterDark, 3) < c
        error('masterDark has fewer channels than the light frame.');
    end

    D = masterDark(:,:,c);

else
    error('masterDark must be 2D or 3D.');
end
end
