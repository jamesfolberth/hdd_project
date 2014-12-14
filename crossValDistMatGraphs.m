
function [correctClassRate, probCorrect,correctClassRate2, probCorrect2,correctClassRate3, probCorrect3] = crossValDistMatGraphs(NumofEvcs)
% Test the classification algorithm following the ideas of Section 6 of
% Dr. Meyer's project guide.
%clc
%close all 
%clear all 
[G1,G2,G3]  = createGraph(.15);

% load(savefile); % load in the distance matrix 'dist'
%distMatknn(dist, 500, 15)
dataDir = getDir();
[wavList,genre] = textread([dataDir,'ground_truth.csv'],'%s %s','delimiter',',');
nSongs = length(wavList);
genre   = strrep(genre, '"', '');

genreValues = getGenres(genre); 
R = cell(10,5); 
R2 = cell(10,5); 
R3 = cell(10,5); 

for n =1: 10
    G = cell(6,5); 
    for i =1:6
        dum =  find(genreValues ==i); 
        pos   = randperm(length(dum));
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
        probCorrect = []; 
        probCorrect2 = []; 
        probCorrect3 = []; 

        confMat = zeros(numel(unique(genreValues))); % 6x6
        confMat2 = zeros(numel(unique(genreValues))); % 6x6
        confMat3 = zeros(numel(unique(genreValues))); % 6x6

        IDXGraph = unnormalizeSpectralCustering(NumofEvcs,G1); 
        IDXGraph2 = normalizeSpectralCustering(NumofEvcs,G1);
        IDXGraph3 = normalizeSpectralCustering2(NumofEvcs,G1);
        
        
%         Predict
%         [predGenre] = multisvm(IDXGraph(trainIndex,:), genreTrain,IDXGraph(testIndex,:));
%          [predGenre2] = multisvm(IDXGraph2(trainIndex,:), genreTrain,IDXGraph2(testIndex,:));
%         [predGenre3] = multisvm(IDXGraph3(trainIndex,:), genreTrain,IDXGraph3(testIndex,:));
    
  K=5;         
           
      for j=1:length(testIndex)
         trueGenre = genreTest(j);
         index = sort ([trainIndex,testIndex(j)]);
            D = G1(index,:);
            D = D(:,index); 
            D1 = G2(index,:);
            D1 = D1(:,index); 
            D2 = G3(index,:);
            D2 = D2(:,index); 
           neighbors = distMatknn1(D, find(index ==testIndex(j)), K);
           neighborGenres = genreTrain(neighbors);
           predGenre = mode(neighborGenres);
         neighbors = distMatknn1(D1, find(index ==testIndex(j)), K);
           neighborGenres = genreTrain(neighbors);
           predGenre2 = mode(neighborGenres);
           neighbors = distMatknn1(D2, find(index ==testIndex(j)), K);
           neighborGenres = genreTrain(neighbors);
           predGenre3 = mode(neighborGenres);
            confMat(predGenre, trueGenre) = confMat(predGenre, trueGenre) + 1;
            confMat2(predGenre2, trueGenre) = confMat(predGenre2, trueGenre) + 1;
            confMat3(predGenre3, trueGenre) = confMat(predGenre3, trueGenre) + 1;

         
%          predGenreTest = predGenre(j);
%          confMat(predGenreTest, trueGenre) = confMat(predGenreTest, trueGenre) + 1;
%          predGenreTest = predGenre2(j);
%          confMat2(predGenreTest, trueGenre) = confMat2(predGenreTest, trueGenre) + 1;
%          predGenreTest = predGenre3(j);
%          confMat3(predGenreTest, trueGenre) = confMat3(predGenreTest, trueGenre) + 1;
      end
        R{n,k} = confMat; 
        R2{n,k} = confMat2;
        R3{n,k} = confMat3;
%         probCorrect = [probCorrect diag( R{n,k})./sum( R{n,k},2)];
%         probCorrect2 = [probCorrect2 diag( R2{n,k})./sum( R2{n,k},2)];
%         probCorrect3 = [probCorrect3 diag( R3{n,k})./sum( R3{n,k},2)];

        
    end
    
end
    crossValAvg = zeros(6,6); 
    crossValSD = zeros(6,6); 
for l =1:3
        if l ==2
            R = R2; 
        elseif l ==3
            R = R3; 
        end

    for i =1:6
        for j =1:6
            
             crossValAvg(i,j)  = round(mean([R{1,1}(i,j),R{1,2}(i,j),R{1,3}(i,j),R{1,4}(i,j),R{1,5}(i,j),...
                                    R{2,1}(i,j),R{2,2}(i,j),R{2,3}(i,j),R{2,4}(i,j),R{2,5}(i,j),...
                                    R{3,1}(i,j),R{3,2}(i,j),R{3,3}(i,j),R{3,4}(i,j),R{3,5}(i,j),...
                                    R{4,1}(i,j),R{4,2}(i,j),R{4,3}(i,j),R{4,4}(i,j),R{4,5}(i,j),...
                                    R{5,1}(i,j),R{5,2}(i,j),R{5,3}(i,j),R{5,4}(i,j),R{5,5}(i,j),...
                                    R{6,1}(i,j),R{6,2}(i,j),R{6,3}(i,j),R{6,4}(i,j),R{6,5}(i,j),...
                                    R{7,1}(i,j),R{7,2}(i,j),R{7,3}(i,j),R{7,4}(i,j),R{7,5}(i,j),...
                                    R{8,1}(i,j),R{8,2}(i,j),R{8,3}(i,j),R{8,4}(i,j),R{8,5}(i,j),...
                                    R{9,1}(i,j),R{9,2}(i,j),R{9,3}(i,j),R{9,4}(i,j),R{9,5}(i,j),...
                                    R{10,1}(i,j),R{10,2}(i,j),R{10,3}(i,j),R{10,4}(i,j),R{10,5}(i,j)]));
                        
        crossValSD(i,j) = std([R{1,1}(i,j),R{1,2}(i,j),R{1,3}(i,j),R{1,4}(i,j),R{1,5}(i,j),...
                                    R{2,1}(i,j),R{2,2}(i,j),R{2,3}(i,j),R{2,4}(i,j),R{2,5}(i,j),...
                                    R{3,1}(i,j),R{3,2}(i,j),R{3,3}(i,j),R{3,4}(i,j),R{3,5}(i,j),...
                                    R{4,1}(i,j),R{4,2}(i,j),R{4,3}(i,j),R{4,4}(i,j),R{4,5}(i,j),...
                                    R{5,1}(i,j),R{5,2}(i,j),R{5,3}(i,j),R{5,4}(i,j),R{5,5}(i,j),...
                                    R{6,1}(i,j),R{6,2}(i,j),R{6,3}(i,j),R{6,4}(i,j),R{6,5}(i,j),...
                                    R{7,1}(i,j),R{7,2}(i,j),R{7,3}(i,j),R{7,4}(i,j),R{7,5}(i,j),...
                                    R{8,1}(i,j),R{8,2}(i,j),R{8,3}(i,j),R{8,4}(i,j),R{8,5}(i,j),...
                                    R{9,1}(i,j),R{9,2}(i,j),R{9,3}(i,j),R{9,4}(i,j),R{9,5}(i,j),...
                                    R{10,1}(i,j),R{10,2}(i,j),R{10,3}(i,j),R{10,4}(i,j),R{10,5}(i,j)]);                        
        end
    end

% latexTable(crossValAvg, 'crossValAvg.tex', '%i', unique(genre));
% latexTable(crossValSD, 'crossValSD.tex', '%3.2f', unique(genre));


% scaled percent correct as done in project guide book


    if l ==2 
        correctClassRate2 = diag(crossValAvg)./reshape(sum(crossValAvg,1), [6 1]);
        probCorrect2 = sum(diag(crossValAvg)./reshape(sum(crossValAvg,1), [6 1])*1/6);
    elseif l ==3 
        correctClassRate3 = diag(crossValAvg)./reshape(sum(crossValAvg,1), [6 1]);
        probCorrect3 = sum(diag(crossValAvg)./reshape(sum(crossValAvg,1), [6 1])*1/6);
    else
        correctClassRate = diag(crossValAvg)./reshape(sum(crossValAvg,1), [6 1]);
        probCorrect = sum(diag(crossValAvg)./reshape(sum(crossValAvg,1), [6 1])*1/6);
    end
end

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
