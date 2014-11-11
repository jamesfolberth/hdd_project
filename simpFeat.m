function [sfeat] = simpFeat(wav, fs, pow, melPowDB, fpMed, opt)
% Compute the simple features described in 2.2.5 of Pampalk 2006
% The features are organized as follows
%
%   sfeat.zcr           = zero crossing rate
%   sfeat.rms           = RMS energy
%   sfeat.noisiness     = perceived noisiness
%   sfeat.avgLoudness   = average loudness (correlated with noisiness)
%   sfeat.percusiveness = s
%   sfeat.specCentroid  = spectral centroid
%   sfeat.fpMax         = maximum fluctuation pattern
%   sfeat.fpSum         = sum of all fluctuation patterns
%   sfeat.fpBass        = bass
%   sfeat.fpAggr        = aggressiveness
%   sfeat.fpDLF         = domination of low frequencies
%   sfeat.fpG           = center of gravity of FP on modulation frequency axis
%   sfeat.fpF           = focus of FP (describes distribution of energy)

sfeat = struct();

% Time domain
sfeat.zcr = sum(abs(diff(sign(wav))))/(2*length(wav)*fs);
sfeat.rms = sqrt(mean(wav.^2));

% Power spectrum
powDB = pow; powDB(powDB<1) = 1; powDB = 10*log10(powDB);
powDBSmooth = zeros(size(powDB,1)-10, floor(size(powDB,2)/10));
for j = 1:floor(size(powDB,2)/10)
   powDBSmooth(:,j) = mean(powDB(11:end, (1:10) + (j-1)*10),2);
end
sfeat.noisiness = sum(sum(abs(diff(powDBSmooth,1))));

% Mel Power Spectrum
sfeat.avgLoudness = mean(melPowDB(:));
%sfeat.percussiveness = mean(abs(diff(melPowDB,1,2)));
sfeat.specCentroid = sum((1:36).*transpose(sum(melPowDB,2))./...
   max(sum(melPowDB(:)),eps(1)));

% Fluctuation patterns
FP = reshape(fpMed, [12 30]);
sfeat.fpMax = max(FP(:));
sfeat.fpSum = sum(FP(:));
sfeat.fpBass = sum(sum(FP(1:2, 3:30)));
sfeat.fpAggr = sum(sum(FP(2:end, 1:4)))/max(sfeat.fpMax,eps(1));
sfeat.fpDLF = sum(sum(FP(1:3,:)))/max(sum(sum(FP(9:12,:))),eps(1));
sfeat.fpG = sum(sum(FP).*(1:30))/max(sfeat.fpSum,eps(1));
sfeat.fpF = mean(FP(:)./max(sfeat.fpMax,eps(1)));

end
