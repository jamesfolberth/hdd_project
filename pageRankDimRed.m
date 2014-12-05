function ranks = pageRankDimRed(feat)
% Takes a training feature matrix and returns a matrix of feature rankings
% Each genre is considered separately to attempt to grab the best features
% for that genre
% The entire collection is then considered to try to find the overall best
% features
% Squared spearman correlation is used for 


if nargin == 0
   %savefile = 'featVecsWCH.mat';
   saveFile = 'featVecsDale.mat';
   load(saveFile, '-mat');

   if ~exist('feat')
      error('Feature matrix ''feat'' not found in savefile: %s',saveFile);
   end
end

% Indices for where the genre switches
% Note: Shouldn't be hardcoded
gInd = [0 320 434 460 505 607 729];

% Initialize the ranking matrix
% Columns correspond to genres
% Rows correspond to features
ranks = zeros(size(feat,1),7);

% For each genre, find the "best" features
for n = 1:6
    rsq = corr(feat(:,gInd(n)+1:gInd(n+1))','type','Spearman').^2;
    rsq(isnan(rsq)) = 0;
    rsq = rsq - eye(size(feat,1));
    
    H = diag(1./sum(rsq,2))*rsq;

    [r,~] = eigs(H',1);
    r = -r;

    [~,I] = sort(r,'descend');
    ranks(:,n) = I;
end

% For the entire collection, find the "best" features
rsq = corr(feat','type','Spearman').^2;
rsq = rsq - eye(size(feat,1));

H = diag(1./sum(rsq,2))*rsq;

[r,~] = eigs(H',1);
r = -r;

[~,I] = sort(r,'descend');
ranks(:,7) = I;

end
