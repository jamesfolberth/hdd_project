dataDir = getDir();

% Load the list of songs
[wavList,genre] = textread([dataDir,'ground_truth.csv'],'%s %s','delimiter',',');
% Fix the names
wavList = strrep( wavList, '"', '');
wavList = strrep( wavList, 'mp3','wav');

nSongs = length(wavList);

wavFile = strcat(dataDir, wavList{1});
if ( ~exist('audioread') )
    [wav,fs] = wavread(wavFile);
else
    [wav,fs] = audioread(wavFile,'double');
end
wav = wav*10^(96/20);
wav = wav(wav~=0);
fTemp = extractFeatures(wav,fs);

featMat = zeros(length(fTemp),nSongs);
featMat(:,1) = fTemp;

for(i = 2:nSongs)
    if( rem(i,20) == 0)
        fprintf(1,'\rSong: %d of %d.',i, nSongs);
    end
    
    wavFile = strcat(dataDir, wavList{i});
    if ( ~exist('audioread') )
        [wav,fs] = wavread(wavFile);
    else
        [wav,fs] = audioread(wavFile,'double');
    end
    wav = wav*10^(96/20);
    wav = wav(wav~=0);
    fTemp = extractFeatures(wav,fs);
    
    featMat(:,i) = fTemp;
end

save('featMat.mat','featMat');