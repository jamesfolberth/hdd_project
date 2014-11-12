function [] = crosValDistMat(savefile)
% Test the classification algorithm following the ideas of Section 6 of
% Dr. Meyer's project guide.

if nargin == 0
   savefile = 'distG1C.mat';
end

load(savefile); % load in the distance matrix 'dist'

distMatknn(dist, 500, 15)




end
