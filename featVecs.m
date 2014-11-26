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
nSongs = length(wavList);
%nSongs = 10; % for testing

printFile = 1; % stdout
%printFile = fopen('/dev/null');

switch opt.method
case 'WCH'

   mfccOpts = struct('segLength',512,'shiftLength',256,...
                     'method','dct','numTerms',20);

   % Compute features
   feat = zeros([86 nSongs]);
   % 01:10   - assorted simple features
   % 11:46   - mean and var of MFCC DCT coeffs 2:19
   % 47:86  - wavelet coeff histogram features (1st 3 moments + energy)

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

      % only store data for the current song
      [melCoeffs, melPowDB, pow] = mfcc(wav, fs, mfccOpts);
      fpMed = flucPat(melPowDB);
      sfeat = simpFeat(wav, fs, pow, melPowDB, fpMed);

      feat(01,i) = sfeat.zcr;
      feat(02,i) = mean(sfeat.percussiveness);
      feat(03,i) = sfeat.specCentroid;

      feat(04,i) = sfeat.fpMax;
      feat(05,i) = sfeat.fpSum;
      feat(06,i) = sfeat.fpBass; 
      feat(07,i) = sfeat.fpAggr;
      feat(08,i) = sfeat.fpDLF;
      feat(09,i) = sfeat.fpG;
      feat(10,i) = sfeat.fpF;

      feat(11:28,i) = mean(melCoeffs(2:19,:),2);
      feat(29:46,i) = var(melCoeffs(2:19,:),0,2);

      feat(47:86,i) = wch(wav);

   end
   fprintf(printFile, '\n');

   save(opt.savefile,'feat');

otherwise
   error(sprintf('Unknown method to make feature vectors: opt.method = %s',...
   opt.method));
end


% if not stdout or stderr, close the file
if printFile > 2
   fclose(printFile);
end

end % featVecs
