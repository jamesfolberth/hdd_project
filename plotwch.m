function [] = plotwch(songIndex)

method = 2;

switch method
case 1
   % pick a random song
   dataDir = getDir();
   [wavList,genre] = textread([dataDir,'ground_truth.csv'],'%s %s','delimiter',',');
   nSongs = length(wavList);
   % Fix the names
   wavList = strrep( wavList, '"', '');
   wavList = strrep( wavList, 'mp3','wav');
   
   if nargin == 0
      songIndex = randi(nSongs);
   end
   
   wavFile = strcat(dataDir, wavList{songIndex});
   
   % read in the song
   if ( isOctave() )
      [wav,fs] = wavread(wavFile);
   else
      [wav,fs] = audioread(wavFile,'double');
   end
   
   wav = wav*10^(96/20);
   
   feat = localwch(wav);
      
   %plot(1:numel(feat), feat)

case 2

   dataDir = getDir();
   [wavList,genre] = textread([dataDir,'ground_truth.csv'],'%s %s','delimiter',',');

   genre = strrep(genre, '_', '\_'); % fix for latex
   %load('featVecsWCH_approx.mat.save','-mat');
   load('featVecsWCH.mat','-mat');

   mode = 'class';
   mode = 'diff';
   switch mode
   case 'class'
      inds = 1:6;
   case 'diff'
      inds = [1 321 435 461 506 608];
   end

   % Standardize feature vectors
   %feat = bsxfun(@minus, feat, mean(feat, 2));
   %feat = bsxfun(@rdivide, feat, var(feat, 0, 2));
   %fprintf(1,'Feature vectors standardized\n');

   for i=1:numel(inds)
      subplot(2,3,i)
      %plot(51:66,feat(51:66,inds(i)),'o');
      plot(51:66,feat(51:66,inds(i))/10^5);
      axis([51 66 0 1])

      %class(char(genre(inds(i))))
      xlabel('WCH Feature index');
      title(sprintf('%s - Track %03d',char(genre(inds(i))), inds(i)));
      %switch mode
      %case 'class'
      %   print(sprintf('Latex/figures/wch_class_%02d.pdf',i),'-dpdf')
      %case 'diff'
      %   print(sprintf('Latex/figures/wch_diff_%02d.pdf',i),'-dpdf')
      %end
   end

   switch mode
   case 'class'
      print(sprintf('Latex/figures/wch_class.pdf'),'-dpdf')
   case 'diff'
      print(sprintf('Latex/figures/wch_diff.pdf'),'-dpdf')
   end

end

end % plotwch

function [wchFeat] = localwch(wav,opt)
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
