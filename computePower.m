function [pow, segLength, shiftLength] = computePower(wav,opt)
% Given WAVE data, compute the power spectrum
% 
% opt - structure containing parameters to use instead of defaults
%    opt.segLength   = length of segments for STFT
%    opt.shiftLength = length of shift (amount of overlap)
%    opt.rebuild     = 1 to rebuild window; don't define to retain window

% Persistent doesnt work with changing segLengths
%persistent wHann

% set parameters
if nargin > 1
   % Length of segments for STFT
   segLength = opt.segLength;
   
   % How much to shift by for each segment
   shiftLength = opt.shiftLength;
else
   segLength = 512; % 46 ms for 11025 Hz sampling frequency
   shiftLength = segLength/2; % 50% overlap
end

% Number of segments
nSegs = floor((length(wav) - segLength)/shiftLength) + 1;

% Construct a segment matrix, with hann window applied to segments
%if ( isempty(wHann) ...
%   || ( nargin > 1 && isfield(opt,'rebuild') && opt.rebuild ) )
   wHann = 0.5*(1-cos(2*pi*(0:segLength-1)/(segLength-1)))';
   wHann = wHann/sum(wHann);
%end
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

changedOpts = 0;

end
