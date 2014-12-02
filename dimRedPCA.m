function [newFeats, trans, explained] = dimRedPCA(feat, perExplained)
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
% http://stats.stackexchange.com/questions/52773/what-can-cause-pca-to-worsen-results-of-a-classifier?lq=1
% --> http://www.win.tue.nl/~mpechen/publications/pubs/PechenizkiyCBMS04.pdf

if nargin == 1
   perExplained = 100;
end

% center data
data = transpose(feat); % data are now rows, variables are columns; convention
mu   = mean(data, 1);
%vars = var(data, 0, 1);
data = bsxfun(@minus, data, mu);
%data = bsxfun(@times, data, 1./sqrt(vars));
%data = data/sqrt(size(data,1)-1); % correction

[U,S,V] = svd(data, 'econ');

latent = diag(S).^2 / (size(data,1)-1);
explained = 100*latent/sum(latent);
%plot(cumsum(explained),'o');
%diag(S)
%error('stop')

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
