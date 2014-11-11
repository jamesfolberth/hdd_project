function [dist] = G1Dist(melCoeffs, opt)
% Return the spectral similarity via a single Gaussian (G1)
% Input:
%    melCoeffs - cell array of MFCC coefficients
%    opt       - options structure
%
% Output:
%    dist - matrix of distances; dist(i,j) is the distance between song i and j

if nargin < 2
   opt = struct('rescFact',450, 'printFile',1);
end

if ~exist('opt.rescFact','var')
   opt.rescFact = 450; % default
end

if ~exist('opt.printFile','var')
   opt.printFile = 1; % stdout
end

nSongs = size(melCoeffs,1);
dist = zeros([nSongs nSongs]);

for(i = 1:(nSongs-1))
   fprintf(opt.printFile,'\rG1DistMat: %d of %d.',i, nSongs-1);
   mi = mean(melCoeffs{i},2);
   coi = cov(melCoeffs{i}');
   icoi = inv(coi);

   for(j = (i+1):nSongs)
      mj = mean(melCoeffs{j},2);
      coj = cov(melCoeffs{j}');
      icoj = inv(coj);
      dist(i,j) = trace(coi*icoj) + trace(coj*icoi) + ...
         trace((icoi+icoj)*(mi-mj)*(mi-mj)'); % K-L divergence
   end
end
fprintf(opt.printFile,'\n');

% rescaling helps with combinations of metrics
dist = -exp(-1/opt.rescFact*dist);
dist = dist + dist';

end
