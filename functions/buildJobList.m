
function jobs = buildJobList(rootDir, ext)
%BUILDJOBLIST Collect light, flat, and dark image jobs.
%
%   jobs = buildJobList(rootDir, ext)
%
%   Expected structure:
%       rootDir/
%       ├── input/
%       │   ├── light/
%       │   ├── flat/
%       │   └── dark/
%       └── output/
%
%   Each job contains:
%       name
%       filePath
%       exists

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

%% Input directories
inputDir = fullfile(rootDir, 'input');

lightDir = fullfile(inputDir, 'light');
flatDir  = fullfile(inputDir, 'flat');
darkDir  = fullfile(inputDir, 'dark');

if ~exist(lightDir, 'dir')
    error('Light directory not found: %s', lightDir);
end

if ~exist(flatDir, 'dir')
    warning('Flat directory not found: %s', flatDir);
end

if ~exist(darkDir, 'dir')
    warning('Dark directory not found: %s', darkDir);
end

%% Collect files
lightFiles = dir(fullfile(lightDir, ['*' ext]));
flatFiles  = dir(fullfile(flatDir,  ['*' ext]));
darkFiles  = dir(fullfile(darkDir,  ['*' ext]));

%% Build job lists
jobs.light = makeJobs(lightFiles, lightDir);
jobs.flat  = makeJobs(flatFiles, flatDir);
jobs.dark  = makeJobs(darkFiles, darkDir);

jobs.rootDir = rootDir;

%% Report
disp(['Light frames found: ' num2str(numel(jobs.light))]);
disp(['Flat frames found:  ' num2str(numel(jobs.flat))]);
disp(['Dark frames found:  ' num2str(numel(jobs.dark))]);

end


function jobList = makeJobs(files, folder)
%MAKEJOBS Convert dir() results into a consistent job structure.

jobList = struct('name', {}, 'filePath', {}, 'exists', {});

for i = 1:numel(files)
    jobList(i).name = files(i).name;
    jobList(i).filePath = fullfile(folder, files(i).name);
    jobList(i).exists = true;
end

end
