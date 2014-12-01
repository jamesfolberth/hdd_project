function [newFeats, trans] = dimRedPCA(feat, perExplained)
% reduce dimension of feature vectors using SVD PCA
% input:
% feat - (num features) x (num songs) data matrix
% perExplained - requested percentage of variance explained
%
% output:
% newFeats - new features
% trans - transformation matrix: old features \to PC space via
%         newFeats = trans * oldFeats
%
% http://arxiv.org/pdf/1404.1100.pdf

if nargin == 1
   perExplained = 100;
end

% standardize data
data = transpose(feat); % data are now rows, variables are columns; convention
mu   = mean(data, 1);
vars = var(data, 0, 1);
data = bsxfun(@minus, data, mu);
data = bsxfun(@times, data, 1./sqrt(vars));
data = data/sqrt(size(data,1)-1); % correction

[U,S,V] = svd(data, 'econ');

latent = diag(S).^2 / (size(data,1)-0);
explained = 100*latent/sum(latent);

% find number of components to use to get to perExplained
accum = 0;
numTerms = numel(explained);
for i=1:numel(explained)
   accum = accum + explained(i);
   if accum >= perExplained - eps(perExplained)
      numTerms = i;
      break
   end
end

trans = transpose(V(:,1:numTerms));
newFeats = trans*feat;

end
