function [] = testflucPat()
% Test flucPat.m, a function to compute various fluctuation patterns

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
[compMelPow, melPowDB] = mfcc(wav, fs, mfccOpt);


[fp] = flucPat(melPowDB);
% Note that we could also do
%  [fp] = flucPat(recoverMelPower(compMelPow,mfccOpt));

% Plot the summary flucutation patterns.  See Figure 2.18 of Pampalk 2006
imagesc(reshape(fp, [12 30]))
colorbar()
set(gca,'YDir','normal')
title('Fluctuation Pattern');
fprintf(1,'This track is "%s".\n', songGenre);

end
