function [feat] = featVecs(wavList, opt)
% compute the WCH features
% Input:
%    wavList - list of tracks (with correct path to file)
%    opt     - options structure
%
% Output:
%    feat    - feature length x number of songs matrix
%

% populate wavList with default tracks
if nargin < 1
   dataDir = getDir();

   % Load the list of songs
   [wavList,genre] = textread([dataDir,'ground_truth.csv'],'%s %s','delimiter',',');
   % Fix the names
   wavList = strrep( wavList, '"', '');
   wavList = strrep( wavList, 'mp3','wav');
end

% populate options structure with default method, etc.
if nargin < 2
   opt = struct('savefile','wchFeatVecs.mat');
end

% Set up variables common to all methods
nSongs = length(wavList);
%nSongs = 10; % for testing

printFile = 1; % stdout
%printFile = fopen('/dev/null');

wchOpts = struct('wName','db8','nLevels',7,'segLength',2^18);

feat = zeros([8*wchOpts.nLevels nSongs]);
for(i = 1:nSongs)
   fprintf(printFile,'\rSong: %d of %d.',i, nSongs);
   wavFile = strcat(dataDir, wavList{i});
   % wavread will be deprecated, so use audioread
   if ( ~exist('audioread') )
      [wav,fs] = wavread(wavFile);
   else
      [wav,fs] = audioread(wavFile,'double');
   end

   wav = wav*10^(96/20);

   [aFeat,dFeat] = wch(wav);
   feat(1:numel(aFeat),i) = aFeat;
   feat(numel(aFeat)+1:end,i) = dFeat;

end
fprintf(printFile, '\n');

save(opt.savefile,'feat');

end
