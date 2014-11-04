function [recovMelPowDB] = recoverMelPower(compMelPow, mfccOpt)
% Recover the Mel power spectrum given the the compressed Mel power spectrum
% and the options structure used to compress the Mel power spectrum.
%
% Example usage:
%    % read in WAVE file to 'wav' and samp. freq. to 'fs'
%    mfccOpt = struct('method','dct','numTerms',20);
%    [compMelPow, melPowDB] = mfcc(wav, fs, mfccOpt);
%    recovMelPowDB = recoverMelPower(compMelPow, mfccOpt);


switch mfccOpt.method
   case 'dct'
      % pad with zeros to recover Mel power spectrum
      recovMelPowDB = [compMelPow; ...
                       zeros([36-mfccOpt.numTerms size(compMelPow,2)])];
      recovMelPowDB = idct(recovMelPowDB);

   case 'wav'
      recovMelPowDB = waverec2(compMelPow{1},compMelPow{2},mfccOpt.wName);

   otherwise
      error(sprintf('Bad MFCC compression option: %s',opt.method));
end

end
