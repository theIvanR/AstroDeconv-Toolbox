% Priors
maxVal_16 = 65535; maxVal_img = 4095;
D = 1500; %LPF Diameter

% Define source and destination directories
srcDir = 'C:\Users\Admin\Desktop\ImProcessor\Input';  % Replace with the actual path to your RAW images
destDir = 'C:\Users\Admin\Desktop\ImProcessor\Output';  % Replace with the destination path for saving TIFF images

% Get list of all RAW files in the source directory
rawFiles = dir(fullfile(srcDir, '*.arw'));  % Change extension if needed
disp(['Raw Files: ', num2str(length(rawFiles))]);

% Low-pass filter function
function filteredImage = applyLowPass(imgTensor, D)
    [M, N, ~] = size(imgTensor);
    [x, y] = meshgrid(1:N, 1:M);
    centerX = ceil(N/2);
    centerY = ceil(M/2);
    lowPassMask = sqrt((x - centerX).^2 + (y - centerY).^2) <= D;
    
    % Function to apply Fourier transform and filter on each channel
    applyFilter = @(channel) abs(ifft2(ifftshift(fftshift(fft2(double(channel))) .* lowPassMask)));
    
    % Apply low-pass filter to R, G, and B channels
    filteredR = applyFilter(imgTensor(:,:,1));
    filteredG = applyFilter(imgTensor(:,:,2));
    filteredB = applyFilter(imgTensor(:,:,3));
    
    % Recombine filtered channels
    filteredImage = cat(3, filteredR, filteredG, filteredB);
end

% Remove Hot Pixel function
function cleanRawImage = removeHotPixelsRAW(cfaImage)
    [M, N] = size(cfaImage);  % Get the dimensions of the RAW image (single-channel)
    
    % Initialize the cleaned RAW image
    cleanRawImage = cfaImage;
    
    % Create shifted versions of the image to get the neighbors
    neighborsUp    = [cfaImage(1,:); cfaImage(1:M-1,:)];      % Shift image down
    neighborsDown  = [cfaImage(2:M,:); cfaImage(M,:)];        % Shift image up
    neighborsLeft  = [cfaImage(:,1), cfaImage(:,1:N-1)];      % Shift image right
    neighborsRight = [cfaImage(:,2:N), cfaImage(:,N)];        % Shift image left
    neighborsUL    = [neighborsUp(:,1), neighborsUp(:,1:N-1)];  % Upper left diagonal
    neighborsUR    = [neighborsUp(:,2:N), neighborsUp(:,N)];    % Upper right diagonal
    neighborsDL    = [neighborsDown(:,1), neighborsDown(:,1:N-1)]; % Lower left diagonal
    neighborsDR    = [neighborsDown(:,2:N), neighborsDown(:,N)];   % Lower right diagonal

    % Average the neighbors for each pixel
    avgNeighbors = (neighborsUp + neighborsDown + neighborsLeft + neighborsRight + ...
                    neighborsUL + neighborsUR + neighborsDL + neighborsDR) / 8;

    % Detect hot pixels by comparing to average neighbors
    hotPixelMask = cfaImage > avgNeighbors * 1.5;

    % Replace hot pixels with the average of their neighbors
    cleanRawImage(hotPixelMask) = avgNeighbors(hotPixelMask);
end


% Loop over each RAW file and process it
tic
parfor i = 1:length(rawFiles)
    % Construct full file path and load image
    rawFilePath = fullfile(rawFiles(i).folder, rawFiles(i).name);
    cfaImage = rawread(rawFilePath);  

    %Step 0: Do RAW Pre Processing
    cfaImage = removeHotPixelsRAW(cfaImage); %remove hot pixels
    colorImage = double(demosaic(cfaImage, 'RGGB')); %demosaic
    %colorImage(:, :, 2) = colorImage(:, :, 2)/2; %prescaler for green, remove for IR!

    % Step 1: Apply low-pass filter (with a chosen cutoff frequency)
    colorImage = applyLowPass(colorImage, D);
    %colorImage = applyLowPass2(colorImage,sigma_s, sigma_r);

    % Scale the image
    colorImage = uint16(maxVal_16 * colorImage / maxVal_img);
    
    % Construct the destination file path (change extension to .tiff)
    [~, fileName, ~] = fileparts(rawFiles(i).name);
    tiffFilePath = fullfile(destDir, [fileName, '_filtered.tiff']);

    % Save the filtered image as a TIFF file
    imwrite(colorImage, tiffFilePath, 'tiff', 'Compression', 'none');  % Optional compression

end

elapsedTime = toc;
disp(['Done! ', num2str(elapsedTime), ' sec']);
