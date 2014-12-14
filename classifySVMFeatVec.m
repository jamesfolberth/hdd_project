function []=classifySVMFeatVec(opt)
% attempt to classify the songs in the test data using the songs in the
% training data
%
% This uses matlab's built-in SVM routines

if nargin < 1 % set all to defaults
   %opt = struct('MCMethod','onevall','dimRed','none');

   %opt = struct('MCMethod','onevall');
   %opt = struct('MCMethod','onevone');
   %opt = struct('MCMethod','ECOC');

   %opt = struct('MCMethod','onevall','dimRed','pr');
   %opt = struct('MCMethod','onevone','dimRed','pr');
   %opt = struct('MCMethod','ECOC','dimRed','pr');

   opt = struct('MCMethod','onevall','dimRed','pr','SVMOrder',2.5,...
      'prDim',125,'prMode','genre0.5');
   %opt = struct('MCMethod','ECOC','dimRed','pr','SVMOrder',2.25,...
   %   'prDim',125,'prMode','genre0.5');

end
% set the needed opts that aren't set to defaults
if ~isfield(opt, 'trainFile')
   opt.trainFile = 'featVecsWCH.mat';
   opt.trainFile = 'featVecsDale.mat';
end
if ~isfield(opt, 'testFile')
   opt.testFile = 'featVecsTestWCH.mat';
   opt.testFile = 'featVecsTestDale.mat';
end
if ~isfield(opt, 'dimRed')
   opt.dimRed = 'none';
end
if ~isfield(opt, 'MCMethod')
   opt.MCMethod = 'onevall';
end
if ~isfield(opt, 'SVMOrder')
   opt.SVMOrder = 2.25;
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
      opt.prDim = 125;
      %opt.prDim = 40;
   end
   if ~isfield(opt, 'prMode')
      %opt.prMode = 'all';
      %opt.prMode = 'genre0';
      opt.prMode = 'genre0.5';
      %opt.prMode = 'genre1';
      %opt.prMode = 'genre2';
   end
   if ~isfield(opt, 'prOpt')
      opt.prOpt = struct('method','basic');
      %opt.prOpt = struct('method','adjusted','factor',0.3);
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
nGenre = numel(unique(trainGenreValues));

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
   if isfield(opt, 'pcaExp')
      [trainFeat, trans, explained] = dimRedPCA(trainFeat, opt.pcaExp);
      testFeat = trans*testFeat;
   elseif isfield(opt, 'pcaNum')
      [trainFeat, trans, explained] = dimRedPCA(trainFeat, 101);
      trainFeat = trainFeat(1:opt.pcaNum,:);
      testFeat = trans*testFeat;
      testFeat = testFeat(1:opt.pcaNum,:);
   else
      error('bad PCA options');
   end

case 'lle'
   nTrain = size(trainFeat,2);
   nTest = size(testFeat,2);
   feat = [trainFeat testFeat];
   [feat] = lle(feat, opt.lleNum, opt.lleDim);
   trainFeat = feat(:,1:nTrain);
   testFeat = feat(:,nTrain+1:nTrain+nTest);

case 'pr'

   [ranks] = pageRankDimRed(trainFeat,opt.prOpt);

   switch opt.prMode
   case 'all' % use ranking based on all tracks
      trainFeat = trainFeat(ranks(1:opt.prDim,7),:);
      testFeat = testFeat(ranks(1:opt.prDim,7),:);

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
      fprintf(1,'Number of dimensions requested = %d\n', opt.prDim);
      fprintf(1,'Number of dimensions used      = %d\n', numel(allRanks));
      trainFeat = trainFeat(allRanks,:);
      testFeat = testFeat(allRanks,:);

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
      %fprintf(1,'Number of dimensions used = %d\n', numel(allRanks));
      trainFeat = trainFeat(allRanks,:);
      testFeat = testFeat(allRanks,:);
     
      %allRanks
      %sort(allRanks,'ascend')

   case 'genre1'
      % Method 1
      % ==========================================
      % find all features, 
      % then weight based on mean rank across genres
      % then take opt.prDim best features
      genreWeights = [1 1 2 4 1 4]; % weights for weighted mean of ranks
      %genreWeights = [1 1 2 3 1 4  1]; % include combined rank

      genreWeights = genreWeights / norm(genreWeights,1);
      selRanks = ranks(1:opt.prDim,1:numel(genreWeights));
      allRanks = unique(selRanks(:));
      rankWeights = zeros(size(allRanks));
      for i = 1:numel(allRanks)
         [I,J] = find(allRanks(i) == selRanks);

         % default rank is max dimension
         fullRanks  = opt.prDim*ones([1 numel(genreWeights)]);
         fullRanks(J) = I;
         rankWeights(i) = dot(genreWeights, fullRanks) / numel(genreWeights);
      end
   
      % low rank weight means high ranking and a good feature
      [~,I] = sort(rankWeights,1,'ascend');
      trainFeat = trainFeat(allRanks(I(1:opt.prDim)),:);
      testFeat = testFeat(allRanks(I(1:opt.prDim)),:);

   case 'genre2'
      % Method 2
      % ==========================================
      % divide top features evenly among genres in some prefered order
      % TODO don't sample evenly.  This may take us too low in the ranking
      %      for the less prefered genres, leading to worse performance
      genrePref = [6 4 5 3 2 1];
      %genrePref = [7 6 4 5 3 2 1];
      prefFeat = zeros([opt.prDim 1]);
      seg = ceil(opt.prDim/numel(genrePref));

      % genre 1
      prefFeat(1:seg) = ranks(1:seg,genrePref(1));

      % genre 2,...,end-1
      for i = 2:numel(genrePref) - 1
         % 'stable' maintains ordering of ranks
         [newFeats] = setdiff(ranks(:,genrePref(i)), prefFeat, 'stable');
         prefFeat(seg*(i-1)+1:seg*i) = newFeats(1:seg);
      end
     
      % last genre
      [newFeats] = setdiff(ranks(:,numel(genrePref)), prefFeat, 'stable');
      prefFeat(seg*(numel(genrePref)-1):numel(prefFeat)) = ...
         newFeats(1:numel(prefFeat)-seg*(numel(genrePref)-1)+1);

      trainFeat = trainFeat(prefFeat,:);
      testFeat = testFeat(prefFeat,:);

      %ranks(1:30,1:6)
      %prefFeat

   otherwise
      error('Unknown PageRank feature selection mode: %s', opt.prMode);
   end
otherwise
   error('Unknown dimension reduction method: %s', opt.dimRed);
end


% train SVM classifiers for this subset
if exist('fitcsvm') 
   switch opt.MCMethod
   case 'onevall'
      mdls = cell([nGenre 1]);
      for g = 1:nGenre
         inds = (trainGenreValues == g);
         mdls{g} = fitcsvm(transpose(trainFeat),inds,...
         'ClassNames',[true false],'Standardize',1,...
         'KernelFunction','polynomial','PolynomialOrder',opt.SVMOrder,...
         'BoxConstraint',10);

         %fprintf(1,'# support vecs = %d of %d\n',nnz(mdls{g}.IsSupportVector),numel(mdls{g}.IsSupportVector)); %sometimes we have lots of support vecs :(
      end

   case 'onevone'
      pairs = nchoosek(1:nGenre,2);
      mdls = cell([size(pairs,1) 1]);
      for g = 1:size(pairs,1)
         inds = (trainGenreValues == pairs(g,1)) | ...
            (trainGenreValues == pairs(g,2));
         mdls{g} = fitcsvm(transpose(trainFeat(:,inds)),...
         trainGenreValues(inds), 'ClassNames', [pairs(g,1) pairs(g,2)],...
         'Standardize',1,'KernelFunction','polynomial',...
         'PolynomialOrder',opt.SVMOrder,'BoxConstraint',10);
         %'Standardize',1,'KernelFunction','linear','BoxConstraint',1);

         %fprintf(1,'# support vecs = %d of %d\n',nnz(mdls{g}.IsSupportVector),numel(mdls{g}.IsSupportVector)); %sometimes we have lots of support vecs :(
      end

   case 'ECOC'
      %C = codeMatrix('hamming1');
      C = codeMatrix('hammingExhaustive');

      mdls = cell([size(C,2) 1]);
      for g = 1:size(C,2)
         inds = zeros([size(trainGenreValues,1) 1]);
         for class=1:size(C,1)
            if C(class,g) == 1
               inds = inds | (trainGenreValues == class);
            end
         end

         mdls{g} = fitcsvm(transpose(trainFeat),...
         inds,...
         'Standardize',1,'KernelFunction','polynomial',...
         'PolynomialOrder',opt.SVMOrder,'BoxConstraint',10);

         %fprintf(1,'# support vecs = %d of %d\n',nnz(mdls{g}.IsSupportVector),numel(mdls{g}.IsSupportVector)); %sometimes we have lots of support vecs :(

      end                               

   otherwise                            
      error('Unknown multiclass method:  %s',opt.MCMethod);
   end                                  
else                                    
   error('Old matlab.');                
end                                     

% make predictions                      
confMat = zeros([nGenre nGenre]); % 6x6
switch opt.MCMethod     
case 'onevall'
   scores = zeros([size(testFeat,2) nGenre]);
   for g = 1:nGenre
      [~,score] = predict(mdls{g}, transpose(testFeat));
      scores(:,g) = score(:,1); % 1st column has positive score
   end
   [~,predGenre] = max(scores,[],2);
  
   %% randomly select predicted genre
   %g = repmat(transpose(1:6), [25 1]);
   %predGenre = g(randperm(150));

   for j = 1:size(testFeat,2)
      trueGenre = testGenreValues(j);
      confMat(predGenre(j), trueGenre) = ...
         confMat(predGenre(j), trueGenre) + 1;
   end

case 'onevone'
   pred = zeros([size(testFeat,2) size(pairs,1)]);
   for g = 1:size(pairs,1)
      pred(:,g) = predict(mdls{g}, transpose(testFeat));
   end
   predGenre = mode(pred,2);

   for j = 1:size(testFeat,2)
      trueGenre = testGenreValues(j);
      confMat(predGenre(j), trueGenre) = ...
         confMat(predGenre(j), trueGenre) + 1;
   end

case 'ECOC'
   codeWord = zeros([size(testFeat,2) size(C,2)]);
   for g = 1:size(C,2)
      codeWord(:,g) = predict(mdls{g}, transpose(testFeat));
   end

   % find min Hamming distance
   dist = zeros([size(testFeat,2) size(C,1)]);
   for cw = 1:size(C,1)
      dist(:,cw) = sum(bsxfun(@ne, codeWord, C(cw,:)),2); % hamming dist
      %nnz(codeWord ~= C(cw,:));
   end
   [~,predGenre] = min(dist,[],2);

   for j = 1:size(testFeat,2)
      trueGenre = testGenreValues(j);
      confMat(predGenre(j), trueGenre) = ...
         confMat(predGenre(j), trueGenre) + 1;
   end

otherwise
   error('Unknown multiclass method: %s',opt.MCMethod);
end

% print results
%latexTable(crossValAvg, 'crossValAvg.tex', '%3.2f', unique(genre));
correctClassRate = diag(confMat)./reshape(sum(confMat,1), [6 1]);

% scaled percent correct as done in project guide book
probCorrect = sum(diag(confMat)./reshape(sum(confMat,1), [6 1])*1/6);

probCorrect
correctClassRate

printClassifyResults(predGenre);

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

function [C] = codeMatrix(name)
% return a binary matrix corresponding to the named error-correcting code
% for six codewords
   switch name
   case 'hamming1'
      C = [0 0 0 0 0 0 ;
           0 1 0 1 0 1 ;
           1 0 0 1 1 0 ;
           1 1 0 0 1 1 ;
           1 1 1 0 0 0 ;
           1 0 1 1 0 1 ];
   
   case 'hammingExhaustive'
      k = 6;
      C = zeros([k 2^(k-1)-1]);
      for level = 1:k
         segLen = 2^(k-level);
         numSegs = 2^(level-1);
   
         for i = 1:numSegs-1
            if mod(i,2) == 0
               C(level,segLen*(i-1)+1:segLen*i) = 1;
            else; 
               C(level,segLen*(i-1)+1:segLen*i) = 0;
            end
         end
   
         C(level,segLen*(numSegs-1)+1:end) = 1;
      end
   
   case 'BCH'
   
   otherwise
      error('Unknown code matrix name %s', name);
   end
end % codeMatrix

function [st] = printClassifyResults(predGenre, filename)
% write our classification results to a file

   if nargin < 2
      filename = 'james_aly_dale_predict.txt';
   end
   
   genreNames = {'classical','electronic','jazz_blues','rock_pop','metal_punk','world'};
   
   try
      fid = fopen(filename,'w+');
   
      if fid == -1
         error('Cannot open/create file %s for writing.',filename);
      end
   
      fprintf(fid, 'Song  Genre\n');
      fprintf(fid, '================\n');
      for i=1:numel(predGenre)
         fprintf(fid, '%03d   %s\n', i, genreNames{predGenre(i)});   
      end
   
      st = fclose(fid);
   
   catch 
      error('Writing to file %s failed',filename);
   end

end
