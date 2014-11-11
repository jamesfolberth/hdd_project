function [] = testSimpFeat()
% Test simpFeat.m, a function to compute various simple features

% pick a random song
dataDir = getDir();
[wavList,genre] = textread([dataDir,'ground_truth.csv'],'%s %s','delimiter',',');
nSongs = length(wavList);
% Fix the names
wavList = strrep( wavList, '"', '');
wavList = strrep( wavList, 'mp3','wav');
songIndex = randi(nSongs);
wavFile = strcat(dataDir, wavList{songIndex});
songGenre = strrep(genre{songIndex}, '"','');

% read in the song
if ( isOctave() )
   [wav,fs] = wavread(wavFile);
else
   [wav,fs] = audioread(wavFile,'double');
end

wav = wav*10^(96/20);

mfccOpt = struct('method','dct','numTerms',20);
%mfccOpt = struct('method','wav','wName','bior3.3','wLevel',5,'numTerms',10);
[compMelPow, melPowDB, pow] = mfcc(wav, fs, mfccOpt);

[fpMed] = flucPat(melPowDB);

simpOpt = struct();
sfeat = simpFeat(wav, fs, pow, melPowDB, fpMed, simpOpt)

end
