function jobs = buildJobList(rootDir, ext)

%% Defaults
if nargin < 1 || isempty(rootDir)
    rootDir = pwd;
end

if nargin < 2 || isempty(ext)
    ext = '.tiff';
end

if ext(1) ~= '.'
    ext = ['.' ext];
end

%% Directories (FIXED)
lightDir = fullfile(rootDir, 'Light');
flatDir  = fullfile(rootDir, 'Flat');

if ~exist(lightDir, 'dir')
    error('Light directory not found: %s', lightDir);
end

if ~exist(flatDir, 'dir')
    warning('Flat directory not found: %s', flatDir);
end

%% Collect files
lightFiles = dir(fullfile(lightDir, ['*' ext]));
flatFiles  = dir(fullfile(flatDir,  ['*' ext]));

%% Build LIGHT list
jobs.light = struct([]);
for i = 1:numel(lightFiles)
    jobs.light(i).name     = lightFiles(i).name;
    jobs.light(i).filePath = fullfile(lightDir, lightFiles(i).name);
    jobs.light(i).exists   = true;
end

%% Build FLAT list
jobs.flat = struct([]);
for i = 1:numel(flatFiles)
    jobs.flat(i).name     = flatFiles(i).name;
    jobs.flat(i).filePath = fullfile(flatDir, flatFiles(i).name);
    jobs.flat(i).exists   = true;
end

jobs.rootDir = rootDir;

end