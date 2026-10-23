
function createAstroDirs(rootDir)
% createAstroDirs Create the required Astro Deconv Toolbox folders.

if ~exist(rootDir, 'dir')
    error('Root directory does not exist: %s', rootDir);
end

inputDir  = fullfile(rootDir, 'input');
outputDir = fullfile(rootDir, 'output');

subDirs = {'light', 'flat', 'dark'};

dirs = [{inputDir}, ...
    cellfun(@(d) fullfile(inputDir, d), subDirs, ...
    'UniformOutput', false), ...
    {outputDir}];

for i = 1:length(dirs)
    if exist(dirs{i}, 'dir')
        disp(['Folder found: ' dirs{i}]);
    else
        mkdir(dirs{i});
        disp(['Folder created: ' dirs{i}]);
    end
end

disp('All directories are ready.');
end
