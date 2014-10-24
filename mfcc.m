function [powCoeffs, melPowDB] = mfcc(wav,fs,opt1,opt2)

%persistent wHann;
persistent melFilter; % TODO profile this to make sure persistent is faster


if( nargin < 3 )
   opt1 = 'dct';
end

% Greatest recoverable frequency
maxFreq = fs/2;

%% Length of segments for STFT
%segLength = 512; % 46 ms for 11025 Hz sampling
%% How much to shift by for each segment
%shiftLength = segLength/2; % 50% overlap
[pow, segLength, shiftLength] = computePower(wav);

% Number of mel frequency bins to use
nMelFreq = 36;
if( isempty(melFilter) )
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
end

% Apply the triangular mel filters to switch to mel power
melPow = melFilter*pow;

% Convert to decibels
melPow(melPow<1) = 1;
melPowDB = 10*log10(melPow);

if( opt1 == 'dct' )
   % Compress the Mel power spectrum
   if( nargin < 4 )
      opt2 = 20;
   end
   nCoeffs = opt2;

   powCoeffs = dct(melPowDB);
   powCoeffs = powCoeffs(1:nCoeffs,:); % TODO should we throw away first term?

elseif( opt1 == 'pca' )
   % Extract the principal components of the Mel power spectrum
   if(nargin < 4)
      opt2 = 3;
   end
 
   melCov = cov(melPowDB');
   [u,s,~] = svd(melCov);
   powCoeffs = u(1:opt2,:)*s;

elseif( opt1 == 'dwt' )
   disp('Wavelet transform not yet implemented!')
end

end
