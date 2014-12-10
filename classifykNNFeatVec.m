function []=classifykNNFeatVec(opt)
% attempt to classify the songs in the test data using the songs in the
% training data
%
% This uses matlab's built-in kNN routines

if nargin < 1 % set all to defaults
   %opt = struct('XValNum', 10, 'kNNNum',5,'dimRed','lle','lleNum',33,'lleDim',20);
   %opt = struct('XValNum', 10, 'kNNNum',5,'dimRed','pca','pcaNum',40);
   opt = struct('XValNum', 10, 'kNNNum',5,'dimRed','none');
   %opt = struct('XValNum', 10, 'kNNNum',5,'dimRed','pr');

end
% set the needed opts that aren't set to defaults
if ~isfield(opt, 'trainFile')
   opt.trainFile = 'featVecsWCH.mat';
end
if ~isfield(opt, 'testFile')
   opt.testFile = 'featVecsTestWCH.mat';
end
if ~isfield(opt, 'kNNNum')
   opt.kNNNum = 5;
end
if ~isfield(opt, 'XValNum')
   opt.XValNum = 10;
end
if strcmp(opt.dimRed, 'lle')
   if ~isfield(opt, 'lleNum')
      opt.lleNum = 37;
   end
   if ~isfield(opt, 'lleDim')
      opt.lleDim = 25;
   end
end
if strcmp(opt.dimRed, 'pca')
   if ~isfield(opt, 'pcaExp')
      opt.pcaExp = 95;
      if ~isfield(opt, 'pcaNum') % no PCA options
         opt.pcaNum = 38;
      end
   end
   if isfield(opt, 'pcaNum')
      opt = rmfield(opt,'pcaExp');
   end
end
if strcmp(opt.dimRed, 'pr')
   if ~isfield(opt, 'prDim')
      %opt.prDim = 67;
      opt.prDim = 118;
      %opt.prDim = 21;
   end
   if ~isfield(opt, 'prMode')
      %opt.prMode = 'all';
      %opt.prMode = 'genre0';
      opt.prMode = 'genre0.5';
      %opt.prMode = 'genre1';
      %opt.prMode = 'genre2';
   end
end

opt

tmp = load(opt.trainFile, 'feat');
trainFeat = tmp.feat;
tmp = load(opt.testFile, 'feat');
testFeat = tmp.feat;

if size(trainFeat,1) ~= size(testFeat,1)
   error('Feature vector lengths do not match.')
end

% get song filenames and genre codes
trainDataDir = getDir();
[trainWavList,trainGenre] = textread([trainDataDir,'ground_truth.csv'],...
   '%s %s','delimiter',',');
nTrain = length(trainWavList);
trainGenre = strrep(trainGenre, '"', '');
trainGenreValues = getGenres(trainGenre);

[testWavList, testGenreValues] = getTestData();

% standardize feature vectors using mean and var from training data
mu = mean(trainFeat, 2);
sd = std(trainFeat, 0, 2);
trainFeat = bsxfun(@minus, trainFeat, mu);
trainFeat = bsxfun(@rdivide, trainFeat, sd);
testFeat = bsxfun(@minus, testFeat, mu);
testFeat = bsxfun(@rdivide, testFeat, sd);
fprintf(1,'Feature vectors standardized\n');

switch opt.dimRed
case 'none'
   % do nothing
case 'pca'
   error('not implemented')
   if isfield(opt, 'pcaExp')
      [feat, trans, explained] = dimRedPCA(feat, opt.pcaExp);
      size(feat,1)
   elseif isfield(opt, 'pcaNum')
      [feat, trans, explained] = dimRedPCA(feat, 101);
      feat = feat(1:opt.pcaNum,:);
   else
      error('bad PCA options');
   end

case 'lle'
   error('not implemented')
   [feat] = lle(feat, opt.lleNum, opt.lleDim);

case 'pr' % XXX this was just copied from crossValSVMFeatVec
          %     this could be outdated; use with caution

   error('not implemented')
   [ranks] = pageRankDimRed(feat);

   switch opt.prMode
   case 'all' % use ranking based on all tracks
      feat = feat(ranks(1:opt.prDim,7),:);

   % use ranking based on genres, and combine up to prDim features
   % TODO how to give preference to certain genres when combining rankings
   %      for onevall, build SVM classifiers based on best features for 
   %        that genre
   %      for onevone, ?
   %      for ECOC, ?
   case 'genre0'
      % Method 0
      % ==========================================
      % Use all the features in the first opt.prDim rows of the rank matrix
      % note that this typically will _not_ use a feature vector of length
      % opt.prDim
      selRanks = ranks(1:opt.prDim,1:6);
      allRanks = unique(selRanks(:));
      fprintf(1,'Number of dimensions used = %d\n', numel(allRanks));
      feat = feat(allRanks,:);

      %allRanks

   case 'genre0.5'
      % Method 0.5
      % ==========================================
      % Use highest ranked features in the order genrePref up to
      % opt.prDim features.  Similar to 'genre0' except we set the number
      % of dimensions and we selected features in order, though this only
      % affects the last few features
      genrePref = [6 4 5 3 2 1];

      % reorderedRanks(:) indexes across the rows of ranks in the order
      % of genrePref
      reorderedRanks = transpose(ranks(:,genrePref));
      allRanks = zeros([opt.prDim 1]);
      numFilled = 0; i = 1;
      while numFilled < opt.prDim
         % try to add feature if not already added
         if ~any(allRanks == reorderedRanks(i))
            numFilled = numFilled + 1;
            allRanks(numFilled) = reorderedRanks(i);
         end

         i = i+1;
         if i > numel(reorderedRanks)
            error('Something terrible has happened');
         end
      end
      %fprintf(1,'Number of dimensions requested = %d\n', opt.prDim);
      %fprintf(1,'Number of dimensions used      = %d\n', numel(allRanks));
      feat = feat(allRanks,:);
     
      allRanks
      %sort(allRanks,'ascend')
   end

otherwise
   error('Unknown dimension reduction method: %s', opt.dimRed);
end

% train kNN classifier for this subset
if( exist('fitcknn') )
    mdl = fitcknn(transpose(trainFeat),trainGenreValues,...
      'NumNeighbors',opt.kNNNum,'Distance','seuclidean');
else
    mdl = ClassificationKNN.fit(transpose(trainFeat),genreTrainValues,...
        'NumNeighbors',opt.kNNNum,'Distance','seuclidean');
end

% make predictions
confMat = zeros(numel(unique(trainGenreValues))); % 6x6
predGenre = predict(mdl, transpose(testFeat));
for j=1:length(testGenreValues)
   trueGenre = testGenreValues(j);
   %predGenre = randi([1 6]); % random prediction ~ 17% correct
   confMat(predGenre(j), trueGenre) = confMat(predGenre(j), trueGenre) + 1;
end

%latexTable(crossValAvg, 'crossValAvg.tex', '%3.2f', unique(genre));

correctClassRate = diag(confMat)./reshape(sum(confMat,1), [6 1]);

% scaled percent correct as done in project guide book
probCorrect = sum(diag(confMat)./reshape(sum(confMat,1), [6 1])*1/6);

probCorrect
correctClassRate

end

function [g, code] = getGenres(genres)
% return a vector of numbers representing the genre of each song in 
% the input genres cell array (entries are strings of genre name)
% optional output is the coding scheme

   genreNames = unique(genres);

   codeCmd = 'code = struct(';
   for i=1:numel(genreNames)-1
      codeCmd = strcat(codeCmd, '''', genreNames{i},''',', sprintf('%d,', i));
   end
   i = numel(genreNames);
   codeCmd = strcat(codeCmd, '''', genreNames{i},''',', sprintf('%d);', i));
   eval(codeCmd);
   %code

   g = zeros([size(genres,1) 1]);
   for i=1:size(genres,1);
      for genreInd=1:numel(genreNames)
         if strcmp(genres{i}, genreNames{genreInd})
            g(i) = getfield(code, genreNames{genreInd});
            break
         end
      end
   end

end % getGenres
