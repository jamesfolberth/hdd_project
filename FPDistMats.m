function [FPMats] = FPDistMats(melCoeffs, mfccOpts, opt)
% Compute  distance matrices for a selection of fluctuation patterns

if nargin < 2
   mfccOpts = struct('method','dct','numTerms',20);
end

if nargin < 3
   opt = struct('method','G1C','printfile',1);
end

if ~exist('opt.printFile','var')
   opt.printFile = 1; % stdout
end

nSongs = size(melCoeffs,1);

switch opt.method
case 'G1C'

   fpMeds = computeFlucPats(melCoeffs, mfccOpts, opt);

   FPMats.FP = zeros([nSongs nSongs]);
   FPMats.FPG = zeros([nSongs nSongs]);
   FPMats.FPBass = zeros([nSongs nSongs]);
   
   for i =1:nSongs-1
      fpMedi = fpMeds(:,i);
      FPi = reshape(fpMedi, [12 30]);
      fpSumi = sum(FPi(:));
      fpGi = sum(sum(FPi).*(1:30))/max(fpSumi,eps(1));
      fpBassi = sum(sum(FPi(1:2, 3:30)));
   
      for j=i+1:nSongs
         %fpMedj = flucPat(recoverMelPower(melCoeffs{j}, mfccOpts));
         fpMedj = fpMeds(:,j);
         FPj = reshape(fpMedj, [12 30]);
         fpSumj = sum(FPj(:));
         fpGj = sum(sum(FPj).*(1:30))/max(fpSumj,eps(1));
         fpBassj = sum(sum(FPj(1:2, 3:30)));
   
         FPMats.FP(i,j) = norm(fpMedi-fpMedj,2);
         FPMats.FPG(i,j) = abs(fpGi-fpGj);
         FPMats.FPBass(i,j) = abs(fpBassi-fpBassj);
      end
   end

   FPMats.FP = FPMats.FP + transpose(FPMats.FP);
   FPMats.FPG = FPMats.FPG + transpose(FPMats.FPG);
   FPMats.FPBass = FPMats.FPBass + transpose(FPMats.FPBass);
   
otherwise
   error('Unknown combination method: opt.method = %s',opt.method);
end

end % FPDistMats

function [fpMeds] = computeFlucPats(melCoeffs, mfccOpts, opt)
% Compute the (median) fluctuation patterns for each track so we don't
% recompute.  Recovering the Mel power spectrum from its compressed form
% is expensive (but otherwise we're looking at a ton of memory usage).

nSongs = size(melCoeffs,1);

% Compute the first fluc. pattern so we can get the proper sizes to 
% initialize fpMeds to store all fluc. patterns.
fpMed = flucPat(recoverMelPower(melCoeffs{1}, mfccOpts)); 
fpMeds = zeros([numel(fpMed) nSongs]);
fpMeds(:,1) = fpMed;
clear fpMed;

for i=2:nSongs
   fprintf(opt.printFile,'\rFPDistMat:computeFlucPats: %d of %d.',i, nSongs);
   fpMeds(:,i) = flucPat(recoverMelPower(melCoeffs{i}, mfccOpts));
end
fprintf(opt.printFile,'\n');

end % computeFlucPats
