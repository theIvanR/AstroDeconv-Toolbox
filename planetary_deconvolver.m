% Debug + robust per-channel Wiener deconvolution
clear; clc; close all;

NSR = 1e-2;

% Load All
y = im2double(imread('y.tif')); fprintf('size(y) = %s\n', mat2str(size(y)));
h = im2double(imread('h.tif'));fprintf('size(h) = %s\n', mat2str(size(h)));
h = im2gray(h);

x_est = zeros(size(y));

%Perform Deconvolution by Channel
for c = 1:size(y,3)
    yc = y(:,:,c);
    if size(h,3) == 1
        hc = h(:,:,1);
    else
        hc = h(:,:,c);
    end

    % Basic checks
    if any(isnan(hc(:))) || any(isinf(hc(:))) || all(hc(:)==0)
        error('Check Data');
    end

    fprintf('chan %d: sum(h)=%.3g, max(h)=%.3g, nnz=%d\n', c, sum(hc(:)), max(hc(:)), nnz(hc));

    % Normalize PSF energy
    hc = hc ./ (sum(hc(:)) + eps);

    % Center PSF: move centroid to image center (important if PSF image has star not centered)
    [rows, cols] = size(hc);
    [X,Y] = meshgrid(1:cols, 1:rows);
    cx = sum(X(:).*hc(:)) / (sum(hc(:))+eps);
    cy = sum(Y(:).*hc(:)) / (sum(hc(:))+eps);
    desired_cx = floor((cols+1)/2);
    desired_cy = floor((rows+1)/2);
    shiftx = round(desired_cx - cx);
    shifty = round(desired_cy - cy);
    hc = circshift(hc, [shifty shiftx]);

    % Show the PSF to verify it's sane
    %figure('Name',sprintf('PSF chan %d',c),'NumberTitle','off'); imagesc(hc); axis image; colorbar;
    %title(sprintf('PSF (chan %d) — centered, normalized',c));

    % Run Semi Wiener deconvolution
    x_est(:,:,c) = deconvwnr(yc, hc, NSR);
end

% Display results using autoscale (helps debugging)
figure('Name','Deconvolution (y | h | x_est)','NumberTitle','off','Color','w');
subplot(1,3,1); imshow(y); title('y (blurry input)');
subplot(1,3,2); imshow(h); title('PSF (as read)');
subplot(1,3,3); imshow(x_est, []); title('x\_est (deconvolved, autoscaled)');

% If you want to save a clipped uint16:
% x_save = im2uint16(mat2gray(x_est)); imwrite(x_save, 'x_est.tif');
