%% Performs a single Guassian (G1) comparison on the set of songs
%%   using MFCC

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

melCoeffs = cell(nSongs,1);

for(i = 1:nSongs)
  fprintf(1,'\rMFCC: %d of %d.',i, nSongs);
  wavFile = strcat(dataDir, wavList{i});
  % wavread will be deprecated, so use audioread
  if ( isOctave() )
     [wav,fs] = wavread(wavFile);
  else
     [wav,fs] = audioread(wavFile,'double');
  end
  
  wav = wav*10^(96/20);
  
  melCoeffs{i} = mfcc(wav,fs,'dct',20);

end
fprintf(1,'\n');

dist = zeros(nSongs);

for(i = 1:(nSongs-1))
  mi = mean(melCoeffs{i},2);
  coi = cov(melCoeffs{i}');
  icoi = inv(coi);
  
  for(j = (i+1):nSongs)
    mj = mean(melCoeffs{j},2);
    coj = cov(melCoeffs{j}');
    icoj = inv(coj);
    dist(i,j) = trace(coi*icoj) + trace(coj*icoi) + ...
      trace((icoi+icoj)*(mi-mj)*(mi-mj)');
  end
end

dist = dist + dist';

toc
