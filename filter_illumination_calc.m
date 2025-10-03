% APS-C sensor: half-diagonal ~14.42 mm
f       = 2000;     % focal length in mm
N       = 10;        % f/5
rs      = 14.42;    % half-diagonal of sensor
d       = 25;       % filter distance from sensor
Dfilter = 31.75;    % 1.25" filter clear aperture

[Dmin, eta] = filter_requirement(f, N, rs, d, Dfilter);

fprintf('Required filter diameter: %.2f mm\n', Dmin);
fprintf('Relative corner illumination: %.2f%%\n', 100*eta);

function [Dmin, eta] = filter_requirement(f, N, rs, d, Dfilter)
%FILTER_REQUIREMENT  Compute minimum filter diameter for no vignetting
%
% Inputs:
%   f       - focal length of telescope (mm)
%   N       - f-ratio (f-number, e.g. 5 for f/5)
%   rs      - half-diagonal of sensor (mm)
%   d       - filter distance from sensor (mm)
%   Dfilter - actual filter clear diameter (mm)
%
% Outputs:
%   Dmin - required filter diameter (mm) for zero vignetting
%   eta  - relative corner illumination [0..1]
%
% Example:
%   % Sony APS-C (diag ~28.84 mm => half-diagonal ~14.42 mm)
%   [Dmin, eta] = filter_requirement(1000, 5, 14.42, 30, 31.75)

    % Cone half-angle from f/ratio
    alpha = atan(1/(2*N));
    % Field angle (corner pixel chief ray)
    theta = atan(rs/f);
    
    % Growth of cone and offset of chief ray at filter plane
    r_cone  = d * tan(alpha);
    r_field = d * tan(theta);
    
    % Minimum half-radius of filter needed
    r_min = rs + r_cone + r_field;
    Dmin  = 2 * r_min;
    
    % Actual half-radius of your filter
    r_filter = Dfilter / 2;
    
    % Relative illumination at sensor corner
    eta = min(1, r_filter / r_min);
end
