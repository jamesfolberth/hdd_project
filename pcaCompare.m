%% Performs PCA analysis of the MFCCs of the songs

clear;

tic;

dataDir = getDir();

% Load the list of songs
[wavList,genre] = textread([dataDir,'ground_truth.csv'],'%s %s','delimiter',',');
nSongs = length(wavList);
% Fix the names
wavList = strrep( wavList, '"', '');
wavList = strrep( wavList, 'mp3','wav');

%nSongs = length(wavList);
nSongs = 50;

principalComponents = cell(nSongs,1);

for(i = 1:nSongs)
  %disp(i);
  %fflush(stdout);
  wavFile = strcat(dataDir, wavList{i});
  % wavread will be deprecated, so use audioread
  if ( isOctave() )
     [wav,fs] = wavread(wavFile);
  else
     [wav,fs] = audioread(wavFile,'double');
  end
  
  wav = wav*10^(96/20);
  
  principalComponents{i} = mfcc(wav,fs,'pca',3);
  
end

dist = zeros(nSongs);

for(i = 1:(nSongs-1))
  for(j = (i+1):nSongs)
    % Need a better distance metric!!!
    dist(i,j) = norm(abs(principalComponents{i}-principalComponents{j}));
  end
end

dist = dist + dist';

toc