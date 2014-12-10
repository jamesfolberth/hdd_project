function [wavList, genreCode] = getTestData()
% this attempts to get the wav list and genre list
% I set up the test data directory as follows:
% test_data/
%    classical/
%    electronic/
%    jazz/
%    punk/
%    rock/
%    world/
%
% returns wavList - cell array of filename strings
%         genreCode - vector genre codes (classical -> 1, electronic -> 2,...)

if isunix()
   [~, userName] = system('whoami');
   userName = strtrim(userName);
elseif ispc()
   [~, userName] = system('echo %USERNAME%');
end

if ( strcmp(userName,'dalekj') )
   error('update test data path')
   dataDir = '/media/removable/SDcard/cd_data/';
elseif ( strcmp(userName,'james') )
   dataDir = './test_data/';
elseif ( strcmp(userName,'alfox') ) 
   error('update test data path')
   dataDir = '/Users/alfox/Documents/Data/';
elseif ( strcmp(deblank(userName),'Dale') )
   error('update test data path')
   dataDir = '../cd_data/';
elseif ( strcmp(userName,'daje3299') )
   error('update test data path')
   dataDir = '';
else
   error('Unknown user name.')
end

% this might be cross-platform...
fileList = getAllFiles(dataDir);

% only keep .wav files
wavIndsc = regexp(fileList,'.wav');
wavInds = [];
for i=1:numel(wavIndsc)
   if ~isempty(wavIndsc{i})
      wavInds = [wavInds; i];
   end
end

wavList = fileList(wavInds);

% find genres
genreCode = zeros([numel(wavList) 1]);
for i=1:numel(wavList)
   if ~isempty(regexp(wavList{i}, 'classical'))
      genreCode(i) = 1;
   elseif ~isempty(regexp(wavList{i}, 'electronic'))
      genreCode(i) = 2;
   elseif ~isempty(regexp(wavList{i}, 'jazz'))
      genreCode(i) = 3;
   elseif ~isempty(regexp(wavList{i}, 'punk'))
      genreCode(i) = 4;
   elseif ~isempty(regexp(wavList{i}, 'rock'))
      genreCode(i) = 5;
   elseif ~isempty(regexp(wavList{i}, 'world'))
      genreCode(i) = 6;
   end
end

if any(genreCode == 0)
   error('James'' regex skillz are weak')
end

end

% stolen from http://stackoverflow.com/questions/2652630/how-to-get-all-files-under-a-specific-directory-in-matlab
% this might be cross-platform...
function fileList = getAllFiles(dirName)

  dirData = dir(dirName);      %# Get the data for the current directory
  dirIndex = [dirData.isdir];  %# Find the index for directories
  fileList = {dirData(~dirIndex).name}';  %'# Get a list of the files
  if ~isempty(fileList)
    fileList = cellfun(@(x) fullfile(dirName,x),...  %# Prepend path to files
                       fileList,'UniformOutput',false);
  end
  subDirs = {dirData(dirIndex).name};  %# Get a list of the subdirectories
  validIndex = ~ismember(subDirs,{'.','..'});  %# Find index of subdirectories
                                               %#   that are not '.' or '..'
  for iDir = find(validIndex)                  %# Loop over valid subdirectories
    nextDir = fullfile(dirName,subDirs{iDir});    %# Get the subdirectory path
    fileList = [fileList; getAllFiles(nextDir)];  %# Recursively call getAllFiles
  end

end
