function [wchFeat] = wch(wav,opt)
% Compute the (Daubechies) Wavelet Coefficient Histogram and return features
%
% wav - vector read of WAVE file data
% opt - options structure
%       opt.wName       - wavelet name (default 'db8')
%       opt.nLevels     - number of levels in decomposition (default 7)
%       opt.segLength   - length of segment (defaul 2^15 ~ 3 seconds)
%

if( nargin < 2 )
   %segLength = 2^15; % ~3s for 11025 Hz sampling
   opt = struct('wName','db8',...
                'nLevels',7,...
                'segLength',2^18); 
                %'segLength',2^16);
end

% trim out the ends of the signal and work with only the middle
%segLength = 2^(nextpow2(length(wav))-1);
%segLength = floor(0.8*length(wav));

if opt.segLength > length(wav)
   opt.segLength = length(wav);
end

startInd = max(floor((length(wav) -opt.segLength)/2), 1);
endInd   = min(length(wav)-startInd, length(wav));
%disp([startInd endInd length(wav)]);

[C,L] = wavedec(wav(startInd:endInd), opt.nLevels, opt.wName);

%%coeffs = detcoef(C,L, 6);
%coeffs = appcoef(C,L, opt.wName, 1);
%[n,x] = hist(coeffs, 60);
%n = n./trapz(x,n); % normalize
%plot(n)
%axis([1 numel(n) 0 0.3*max(n)]);
%%error('stuff')

%subbands = 1:7;
subbands = 4:7; % first few don't look useful
wchFeat = zeros([4*numel(subbands) 1]);
for i=1:numel(subbands)
   dcoeffs = detcoef(C,L, subbands(i)); 
   %acoeffs = appcoef(C,L, opt.wName, subbands(i)); % TODO should we use app coeffs?
   [dn] = hist(dcoeffs, 60); % compute WCH
   %[an] = hist(acoeffs, 60); 
   inds = 1:60; % following Li et al. '03, we think of n going from 1 to 60
   dn = dn./trapz(inds,dn); % normalize
   %an = an./trapz(inds,an);

   %plot(inds,dn)
   %pause(1)

   % moments
   wchFeat(4*(i-1)+1) = sum(inds.*dn);
   wchFeat(4*(i-1)+2) = sum(inds.^2.*dn);
   wchFeat(4*(i-1)+3) = sum(inds.^3.*dn);

   %wchFeat(8*(i-1)+5) = sum(inds.*an);
   %wchFeat(8*(i-1)+6) = sum(inds.^2.*an);
   %wchFeat(8*(i-1)+7) = sum(inds.^3.*an);

   % subband energy
   wchFeat(4*(i-1)+4) = mean(abs(dcoeffs));
   %wchFeat(8*(i-1)+8) = mean(abs(acoeffs));

end

end
