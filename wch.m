function [wchFeat] = wch(wav,fs,opt)
% Compute the (Daubechies) Wavelet Coefficient Histogram and return features
%
% wav - vector read of WAVE file data
% fs  - sampling frequency
% opt - options structure
%       opt.wName       - wavelet name (default 'db8')
%       opt.nLevels     - number of levels in decomposition (default 7)
%       opt.segLength   - length of segment (defaul 2^15 ~ 3 seconds)
%

if( nargin < 3 )
   %segLength = 2^15; % ~3s for 11025 Hz sampling
   opt = struct('wName','db8',...
                'nLevels',7,...
                'segLength',2^15);
end

% trim out the ends of the signal and work with only the middle
%segLength = 2^(nextpow2(length(wav))-1);
%segLength = floor(0.8*length(wav));
segLength = 2^16;
startInd = floor((length(wav) -segLength)/2);
endInd   = length(wav)-startInd;

[C,L] = wavedec(wav(startInd:endInd), opt.nLevels, opt.wName);

%coeffs = detcoef(C,L, 6);
%%coeffs = appcoef(C,L, opt.wName, 1);
%[n,x] = hist(coeffs, 60);
%n = n./trapz(x,n); % normalize
%plot(n)
%axis([1 numel(n) 0 0.3*max(n)]);

%subbands = 1:7;
subbands = 3:7; % first few don't look useful
wchFeat = zeros([4*numel(subbands) 1]);
for i=1:numel(subbands)
   coeffs = detcoef(C,L, subbands(i)); % TODO should we also use approx coeffs
   [n] = hist(coeffs, 60); % compute WCH
   inds = 1:60; % following Li et al. '03, we think of n going from 1 to 60
   n = n./trapz(inds,n); % normalize
   %plot(n); pause(1)

   % moments
   wchFeat(4*(i-1)+1) = sum(inds.*n);
   wchFeat(4*(i-1)+2) = sum(inds.^2.*n);
   wchFeat(4*(i-1)+3) = sum(inds.^3.*n);

   % subband energy
   wchFeat(4*(i-1)+4) = mean(abs(coeffs));

end

end
