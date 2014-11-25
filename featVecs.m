function [feat] = featVecs(wavList, opt)
% combine various features to create a matrix of feature vectors
% Input:
%    wavList - list of tracks (with correct path to file)
%    opt     - options structure
%
% Output:
%    feat    - feature length x number of songs matrix

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
   opt = struct('method','WCH','savefile','featVecsWCH.mat');
   %opt = struct('method','superAwesomeMethod','savefile','featVecsSAM.mat');
end

% Set up variables common to all methods
%nSongs = length(wavList);
nSongs = 10; % for testing

printFile = 1; % stdout
%printFile = fopen('/dev/null');

switch opt.method
case 'WCH'

   mfccOpts = struct('method','dct','numTerms',20);

   % Compute features
   feat = zeros([20+0 nSongs]);
   % 1:20 - wavlet coeff histogram features (1st 3 moments + energy)
   %
   %

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

      melCoeffs = mfcc(wav,fs,mfccOpts); % only store the current MFCCs

      feat(1:20,i) = wch(wav);

   end
   fprintf(printFile, '\n');

   save(opt.savefile,'feat');

otherwise
   error(sprintf('Unknown method to combine metrics: opt.method = %s',...
   opt.method));
end


% if not stdout or stderr, close the file
if printFile > 2
   fclose(printFile);
end

end % featVecs
