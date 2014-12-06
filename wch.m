function [aFeat,dFeat] = wch(wav,opt)
% Compute the Wavelet Coefficient Histogram and return features
%
% wav - vector read of WAVE file data
% opt - options structure
%       opt.wName       - wavelet name (default 'db8')
%       opt.nLevels     - number of levels in decomposition (default 7)
%       opt.segLength   - length of segment (default 2^18 ~23 seconds)
%
% db8 - Daubechies 8 tap wavelet
%  + has a fast transform
%  + widely used
%  - Dr. Meyer has reservations
% 
%
% bior4.4 - Biorthogonal
%  + use with swt as described by Dr. Meyer
%
% swt vs. dwt
%  + swt is time-invariant, while dwt is not time-invariant
%

%warning('James has changed outputs of wch.m.');

if( nargin < 2 )
   opt = struct('wName','bior4.4',...
                'nLevels',7,...
                'segLength',2^18); 
                %'segLength',2^16);
end

% use the middle opt.segLength samples from the song

% to use the discrete stationary wavelet transform, the signal length
% must be divisible by 2^opt.nLevels
if opt.segLength > length(wav)
   opt.segLength = 2^(nextpow2(length(wav))-1);
end

% Okay for dwt, but not okay for swt
startInd = max(floor((length(wav) -opt.segLength)/2), 1);
endInd   = min(startInd + opt.segLength-1, length(wav));
%disp([startInd endInd length(wav)]);

% DEPRECATED
%if endInd-startInd+1 ~= opt.segLength
%   warning('selecting smaller segment.');
%   np = nextpow2(endInd-startInd+1)-1;
%   startInd = floor((length(wav) - 2^np)/2);
%   endInd   = startInd + 2^np - 1;
%   numel(wav(startInd:endInd))
%end

% use the stationary wavelet transform
% returns matrix of approx and detail coefficients
[A,D] = mywavedec(wav(startInd:endInd), opt.nLevels, opt.wName);

nBins = 60;

aFeat = zeros([4*opt.nLevels 1]);
dFeat = zeros([4*opt.nLevels 1]);
inds = 1:nBins;
for i=1:opt.nLevels
   dcoeffs = D(i,:);
   acoeffs = A(i,:);
   [dn] = hist(dcoeffs, nBins); % compute WCH
   [an] = hist(acoeffs, nBins); 
   inds = 1:60; % just pick some bins
   dn = dn./trapz(inds,dn); % normalize
   an = an./trapz(inds,an);

   %plot(inds,dn)
   %pause(1)

   % moments
   aFeat(4*(i-1)+1) = sum(inds.*an);
   aFeat(4*(i-1)+2) = sum(inds.^2.*an);
   aFeat(4*(i-1)+3) = sum(inds.^3.*an);
 
   dFeat(4*(i-1)+1) = sum(inds.*dn);
   dFeat(4*(i-1)+2) = sum(inds.^2.*dn);
   dFeat(4*(i-1)+3) = sum(inds.^3.*dn);
 
   % subband energy
   aFeat(4*(i-1)+4) = mean(abs(acoeffs));
   dFeat(4*(i-1)+4) = mean(abs(dcoeffs));

end

end

function [A,D] = mywavedec(x,n,IN3,IN4)
% multi-level stationary wavelet decomposition using the method explained by
% Dr. Meyer

% Check arguments.
if nargin==3
    [Lo_D,Hi_D] = wfilters(IN3,'d');
else
    Lo_D = IN3;   Hi_D = IN4;
end

% Initialization.
s = size(x); x = x(:)'; % row vector
if isempty(x) , return; end

[A,D] = swt(x,n,Lo_D,Hi_D);

end
