function [feat] = featVecsAppend(savefile, wavList)
% append WCH features to the feature vector matrix contained in 'savefile'
% note that some savefiles already have WCH featurs, so we'll just overwrite them

if nargin < 1
   % this is the reference copy of Dale's features
   savefile = 'featVecsDale.PUSHmat';
end

if nargin < 2
   dataDir = getDir();
   % Load the list of songs
   [wavList,genre] = textread([dataDir,'ground_truth.csv'],'%s %s','delimiter',',');
   % Fix the names
   wavList = strrep( wavList, '"', '');
   wavList = strrep( wavList, 'mp3','wav');
end

load(savefile, '-mat');

if ~exist('feat')
   error('Feature vector matrix ''feat'' not found in savefile %s.', savefile);
end

oldFeat = feat; clear feat;
[nFeats,nSongs] = size(oldFeat);

if nSongs ~= numel(wavList)
   error('Number of songs in ''feat'' and ''wavList'' don'' match.');
end

printFile = 1; % stdout
%printFile = fopen('/dev/null');

% working with Dale's reference features
if nFeats == 198
   
   wchOpts = struct('wName','bior4.4','nLevels',7,'segLength',2^18);
   
   % copy non-WCH features first
   feat = zeros([182 + wchOpts.nLevels*8 nSongs]);
   feat(1:182,:) = oldFeat(1:182,:);

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

      feat(182+1:182+wchOpts.nLevels*4,i) = aFeat; 
      feat(182+1+wchOpts.nLevels*4:182+wchOpts.nLevels*8,i) = dFeat;
   
   end
   fprintf(printFile, '\n');
   
   save('featVecsAppend.mat','feat');

else
   error('Not implemented to support other feature matrices.');
end

end
