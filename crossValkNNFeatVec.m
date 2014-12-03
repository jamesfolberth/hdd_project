function [crossValAvg,crossValSD,probCorrect]=crossValkNNFeatVec(savefile,opt)
% Test the classification algorithm following the ideas of Section 6 of
% Dr. Meyer's project guide.
%
% This uses matlab's built-in kNN routines

if nargin < 1
   savefile = 'featVecsWCH.mat';
   %savefile = 'featVecsDale.mat';
end 

if nargin < 2 % set all to defaults
   %opt = struct('XValNum', 10, 'kNNNum',5,'dimRed','lle','lleNum',33,'lleDim',20);
   %opt = struct('XValNum', 10, 'kNNNum',5,'dimRed','pca','pcaNum',40);
   opt = struct('XValNum', 10, 'kNNNum',5,'dimRed','none');

else % set the needed opts that aren't set to defaults
   if ~isfield(opt, 'kNNNum')
      opt.kNNNum = 5;
   end
   if ~isfield(opt, 'XValNum')
      opt.XValNum = 10;
   end
   if isfield(opt, 'lle')
      if ~isfield(opt, 'lleNum')
         opt.lleNum = 37;
      end
      if ~isfield(opt, 'lleDim')
         opt.lleDim = 25;
      end
   end
   if isfield(opt, 'pca')
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
end

load(savefile);
dataDir = getDir();
[wavList,genre] = textread([dataDir,'ground_truth.csv'],'%s %s','delimiter',',');
nSongs = length(wavList);
genre   = strrep(genre, '"', '');
genreValues = getGenres(genre);

% standardize feature vectors
feat = bsxfun(@minus, feat, mean(feat, 2));
feat = bsxfun(@rdivide, feat, var(feat, 0, 2));
fprintf(1,'Feature vectors standardized\n');

switch opt.dimRed
case 'none'
   % do nothing
case 'pca'
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
   [feat] = lle(feat, opt.lleNum, opt.lleDim);
otherwise
   error('Unknown dimension reduction method: %s', opt.dimRed);
end

%feat = bsxfun(@minus, feat, mean(feat, 2));
%feat = bsxfun(@rdivide, feat, var(feat, 0, 2));
%fprintf(1,'Reduced feature vectors standardized\n');

% Begin cross validation
R = cell(opt.XValNum,5); 
for n =1:size(R,1);
   fprintf(1,'\rn = %d',n);
   G = cell(6,5); 
   for i =1:6
      dum = find(genreValues ==i); 
      pos = randperm(length(dum));
      for l = 1:5
         if l ==5
            G{i,l} = dum(pos(1+floor((l-1)*length(dum)/5): end))'; 
         else
            G{i,l} = dum(pos(1+floor((l-1)*length(dum)/5): floor((l)*length(dum)/5)))'; 
         end
      end
   end

   for k = 1:5
      testIndex = [ G{1,k},G{2,k},G{3,k},G{4,k},G{5,k},G{6,k}] ; 
      trainIndex =setdiff([1:nSongs], unique(testIndex)); 
      genreTest = genreValues(testIndex);
      genreTrain = genreValues(trainIndex);

      % train kNN classifier for this subset
      if( exist('fitcknn') )
          mdl = fitcknn(transpose(feat(:,trainIndex)),genreTrain,...
            'NumNeighbors',opt.kNNNum,'Distance','seuclidean');
      else
          mdl = ClassificationKNN.fit(transpose(feat(:,trainIndex)),genreTrain,...
              'NumNeighbors',opt.kNNNum,'Distance','seuclidean');
      end

      % make predictions
      confMat = zeros(numel(unique(genreValues))); % 6x6
      predGenre = predict(mdl, transpose(feat(:,testIndex)));
      for j=1:length(testIndex)
         trueGenre = genreTest(j);
         %predGenre = randi([1 6]); % random prediction ~ 17% correct
         confMat(predGenre(j), trueGenre) = confMat(predGenre(j), trueGenre) + 1;
      end
      R{n,k} = confMat; 

   end

end
fprintf(1,'\n');
crossValAvg = zeros(6,6); 
crossValSD = zeros(6,6); 

% compute mean and std dev of confusion matrixes
accum = zeros([5*size(R,1) 1]);
for i =1:6
   for j =1:6
      for row=1:size(R,1);
         accum(5*(row-1)+1:5*row) = [R{row,1}(i,j); R{row,2}(i,j); ...
            R{row,3}(i,j); R{row,4}(i,j); R{row,5}(i,j)];
      end
      crossValAvg(i,j) = round(mean(accum));
      crossValSD(i,j)  = std(accum);
   end
end
latexTable(crossValAvg, 'crossValAvg.tex', '%i', unique(genre));
latexTable(crossValSD, 'crossValSD.tex', '%3.2f', unique(genre));

correctClassRate = diag(crossValAvg)./reshape(sum(crossValAvg,1), [6 1]);

% scaled percent correct as done in project guide book
probCorrect = sum(diag(crossValAvg)./reshape(sum(crossValAvg,1), [6 1])*1/6);

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
