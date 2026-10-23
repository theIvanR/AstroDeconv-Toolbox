
%% 0: Initialize toolbox and directory structure

rootDir = pwd;

addpath(genpath(fullfile(rootDir, 'functions')));
createAstroDirs(rootDir);

inputDir  = fullfile(rootDir, 'input');
outDir    = fullfile(rootDir, 'output');

%% 1: Construct jobs and execute

% Ensure all input files have been converted to TIFF via the auxiliary script.
jobs = buildJobList(rootDir, '.tiff');

% 1.1 Hard check for light frames
if isempty(jobs.light)
    error('No LIGHT frames found in: %s', inputDir);
end

% 1.2 Soft check for flat frames
masterFlat = [];

if isempty(jobs.flat)
    warning('No FLAT frames found. Skipping flat correction.');
else
    masterFlat = make_master_flat(jobs.flat);
end

% 1.3 Soft check for dark frames
masterDark = [];

if isempty(jobs.dark)
    warning('No DARK frames found. Skipping dark correction.');
else
    masterDark = make_master_dark(jobs.dark);
end

%% 2: Analyze

net = denoisingNetwork("DnCNN");

for i = 1:numel(jobs.light)

    inFile = jobs.light(i).filePath;
    [~, baseName, ~] = fileparts(jobs.light(i).name);
    outFile = fullfile(outDir, [baseName '.tiff']);

    fprintf('Processing %s (%d/%d)\n', ...
        jobs.light(i).name, i, numel(jobs.light));

    input_image = im2single(imread(inFile));
    
    output_image = zeros(size(input_image), 'like', input_image);

    % 2.1 Remove impulse outliers and apply flat correction per channel
    parfor c = 1:size(input_image, 3)

        chan = input_image(:, :, c);

        % 1: Dark subtraction
        if ~isempty(masterDark)
            D = getDarkChannel(masterDark, c);
            chan = chan - D;
        end

        % 2 Remove noise impulses
        chan = removeImpulseOutliers(chan);

        % 3 Flat divide
        if ~isempty(masterFlat)
            F = getFlatChannel(masterFlat, c);
            F = max(F, eps('single'));
            chan = chan ./ F;
        end

        output_image(:, :, c) = chan;
    end

    % 2.2 Remove color impulse outliers
    output_image = removeColorImpulseOutliers(output_image);

    % Save processed image
    if size(output_image, 3) == 1
        output_image = output_image(:, :, 1);
    end

    imwrite(im2uint16(output_image), outFile, 'Compression', 'lzw');

    fprintf('Saved: %s\n', outFile);

end

disp('All light frames processed successfully.');