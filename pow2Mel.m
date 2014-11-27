function melPowDB = pow2Mel(pow,fs,opt)

if(nargin < 2)
    opt = struct('segLength', 2046, ...
                 'shiftLength', 1028);
end

maxFreq = fs/2;

nMelFreq = 36;

% current frequency bins
freq = linspace(0,maxFreq,opt.segLength/2 + 1);

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
melFilter = zeros(nMelFreq,opt.segLength/2+1);
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

end