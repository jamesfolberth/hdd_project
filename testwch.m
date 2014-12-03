function [] = testwch(songIndex)
% Function used to draw a song and run various tests with wch.m

% pick a random song
dataDir = getDir();
[wavList,genre] = textread([dataDir,'ground_truth.csv'],'%s %s','delimiter',',');
nSongs = length(wavList);
% Fix the names
wavList = strrep( wavList, '"', '');
wavList = strrep( wavList, 'mp3','wav');

if nargin == 0
   songIndex = randi(nSongs);
end

wavFile = strcat(dataDir, wavList{songIndex});

% read in the song
if ( isOctave() )
   [wav,fs] = wavread(wavFile);
else
   [wav,fs] = audioread(wavFile,'double');
end

fs
2^18/fs

%wav = wav*10^(96/20);

feat = wch(wav)

plot(1:numel(feat), feat)

end
