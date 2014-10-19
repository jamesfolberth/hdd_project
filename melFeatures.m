%% Code to extract Mel Frequency Cepstral Coefficients
%% currently extracts coefficients for the full length of a random song

clear;

% Base directory for all of the data
% Should not be hardcoded in here as it is different for everyone
%dataDir = '/media/removable/SDcard/cd_data/';
dataDir = './cd_data/';

% Load the list of songs
[wavList,genre] = textread([dataDir,'ground_truth.csv'],'%s %s','delimiter',',');
nSongs = length(wavList);
% Fix the names
wavList = strrep( wavList, '"', '');
wavList = strrep( wavList, 'mp3','wav');

% Load a song, currently randomly
% Eventually will loop over all of the songs
wavFile = strcat(dataDir, wavList{randi(nSongs)});
% wavread will be deprecated, so use audioread
[wav,fs] = audioread(wavFile,'double');

% wavread returns values between -1 and 1
% rescale to correspond to a max being 96 dB
wav = wav*10^(96/20);

% Greatest recoverable frequency
maxFreq = fs/2;

% Length of segments for STFT
segLength = 512; % 46 ms for 11025 Hz sampling
% How much to shift by for each segment
shiftLength = segLength/2; % 50% overlap

% Number of segments
nSegs = floor((length(wav) - 512)/256) + 1;
% Construct a segment matrix, with hann window applied to segments
wHann = 0.5*(1-cos(2*pi*(0:segLength-1)/(segLength-1)))';
wHann = wHann/sum(wHann);
segMat = zeros(segLength,nSegs);
for n = 1:nSegs
  ind = (1:segLength) + shiftLength*(n-1);
  segMat(:,n) = wHann.*wav(ind);
end

% Calculate the power of the segments
%pow = abs(1/sqrt(segLength) * fft(segMat)).^2;
pow = abs(fft(segMat,[],1)).^2;
% Keep only the first half (+1) of the elements
pow = pow(1:(segLength/2 + 1),:);

% NOTE: Centers and heights of the filters needs to be done only once
%   The calculation should be done outside of the song loop!
% Number of mel frequency bins to use
nMelFreq = 36;
% current frequency bins
freq = linspace(0,maxFreq,segLength/2 + 1);
% mel frequency bins
melFreq = 1127.01048*log(1 + freq/700);
melFreqIdx = linspace(0,melFreq(end),nMelFreq + 2);

% Grab the indices of the edges of the filters
freqInd = zeros(nMelFreq + 2,1);
for n = 1:(nMelFreq + 2)
  [~, freqInd(n)] = min(abs(melFreq - melFreqIdx(n)));
end

% Center and height of the triangular filters
filterCenter = freq(freqInd);
filterHeight = 2 ./ (filterCenter(3:nMelFreq+2)-filterCenter(1:nMelFreq));

% Construct a filter matrix
melFilter = zeros(nMelFreq,segLength/2+1);
for n = 1:nMelFreq
  melFilter(n,:) = (freq > filterCenter(n) & freq <= filterCenter(n+1)) .* ...
    filterHeight(n).*(freq - filterCenter(n))/(filterCenter(n+1)-filterCenter(n)) + ...
    (freq > filterCenter(n+1) & freq <= filterCenter(n+2)) .* ...
    filterHeight(n).*(filterCenter(n+2) - freq)/(filterCenter(n+2)-filterCenter(n+1));
end

% Apply the triangular mel filters to switch to mel power
melPow = melFilter*pow;

% Convert to decibels
melPow(melPow<1) = 1;
melPowDB = 10*log10(melPow);

% Compress the Mel power spectrum
% NOTE: Currently using DCT, however PCA or wavelets might be useful
nCoeffs = 20;

powCoeffs = dct(melPowDB);
powCoeffs = powCoeffs(1:nCoeffs,:);
