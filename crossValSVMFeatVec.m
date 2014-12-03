function [crossValAvg,crossValSD,probCorrect]=crossValSVMFeatVec(savefile,opt)
% Test the classification algorithm following the ideas of Section 6 of
% Dr. Meyer's project guide.
%
% This uses matlab's built-in SVM routines

if nargin < 1
   savefile = 'featVecsWCH.mat';
   %savefile = 'featVecsDale.mat';
end 

if nargin < 2 % set to default values
   % MCMethod - multiclass method ('onevall','onevone''ECOC')
   %opt = struct('MCMethod','onevall');
   %opt = struct('MCMethod','onevone');
   opt = struct('MCMethod','ECOC');
end

% populate needed options that aren't set with default values
if ~isfield(opt, 'dimRed')
   opt.dimRed = 'none';
end
if ~isfield(opt, 'MCMethod')
   opt.MCMethod = 'onevall';
end
if ~isfield(opt, 'SVMOrder')
   opt.SVMOrder = 2.25;
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

load(savefile);
dataDir = getDir();
[wavList,genre] = textread([dataDir,'ground_truth.csv'],'%s %s','delimiter',',');
nSongs = length(wavList);
genre   = strrep(genre, '"', '');
genreValues = getGenres(genre);
genreNames = unique(genre);

% Standardize feature vectors
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
      [feat, trans, explained] = dimRedPCA(feat, 101); % all singular vals
      feat = feat(1:opt.pcaNum,:);
   else
      error('bad PCA options');
   end

case 'lle'
   [feat] = lle(feat, opt.lleNum, opt.lleDim);
otherwise
   error('Unknown dimension reduction method: %s', opt.dimRed);
end


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
      trainIndex = setdiff([1:nSongs], unique(testIndex)); 
      genreTest = genreValues(testIndex);
      genreTrain = genreValues(trainIndex); 

      % train SVM classifiers for this subset
      if exist('fitcsvm') 
         switch opt.MCMethod
         case 'onevall'
            mdls = cell([6 1]);
            for g = 1:numel(genreNames)
               inds = (genreTrain == g);
               mdls{g} = fitcsvm(transpose(feat(:,trainIndex)),inds,...
               'ClassNames',[true false],'Standardize',1,...
               'KernelFunction','polynomial','PolynomialOrder',opt.SVMOrder,...
               'BoxConstraint',10);

               %fprintf(1,'# support vecs = %d of %d\n',nnz(mdls{g}.IsSupportVector),numel(mdls{g}.IsSupportVector)); %sometimes we have lots of support vecs :(
            end

         case 'onevone'
            pairs = nchoosek(1:numel(genreNames),2);
            mdls = cell([size(pairs,1) 1]);
            for g = 1:size(pairs,1)
               inds = (genreTrain == pairs(g,1)) | (genreTrain == pairs(g,2));
               mdls{g} = fitcsvm(transpose(feat(:,trainIndex(inds))),...
               genreTrain(inds), 'ClassNames', [pairs(g,1) pairs(g,2)],...
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
               inds = zeros([size(genreTrain,1) 1]);
               for class=1:size(C,1)
                  if C(class,g) == 1
                     inds = inds | (genreTrain == class);
                  end
               end

               mdls{g} = fitcsvm(transpose(feat(:,trainIndex)),...
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
      confMat = zeros(numel(unique(genreValues))); % 6x6
      switch opt.MCMethod     
      case 'onevall'
         scores = zeros([numel(testIndex) numel(genreNames)]);
         for g = 1:numel(genreNames)
            [~,score] = predict(mdls{g}, transpose(feat(:,testIndex)));
            scores(:,g) = score(:,1); % 1st column has positive score
         end
         [~,predGenre] = max(scores,[],2);

         for j = 1:numel(testIndex)
            trueGenre = genreTest(j);
            confMat(predGenre(j), trueGenre) = ...
            confMat(predGenre(j), trueGenre) + 1;
         end

      case 'onevone'
         pred = zeros([numel(testIndex) size(pairs,1)]);
         for g = 1:size(pairs,1)
            pred(:,g) = predict(mdls{g}, transpose(feat(:,testIndex)));
         end
         predGenre = mode(pred,2);

         for j = 1:numel(testIndex)
            trueGenre = genreTest(j);
            confMat(predGenre(j), trueGenre) = ...
            confMat(predGenre(j), trueGenre) + 1;
         end

      case 'ECOC'
         codeWord = zeros([numel(testIndex) size(C,2)]);
         for g = 1:size(C,2)
            codeWord(:,g) = predict(mdls{g}, transpose(feat(:,testIndex)));
         end

         % find min Hamming distance
         dist = zeros([numel(testIndex) size(C,1)]);
         for cw = 1:size(C,1)
            dist(:,cw) = sum(bsxfun(@ne, codeWord, C(cw,:)),2); % hamming dist
            %nnz(codeWord ~= C(cw,:));
         end
         [~,predGenre] = min(dist,[],2);

         for j = 1:numel(testIndex)
            trueGenre = genreTest(j);
            confMat(predGenre(j), trueGenre) = ...
            confMat(predGenre(j), trueGenre) + 1;
         end

      otherwise
         error('Unknown multiclass method: %s',opt.MCMethod);
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
