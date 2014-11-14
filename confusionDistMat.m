function [confMat] = confusionDistMat(savefile, opt)
% Test the classification algorithm following the ideas of Section 6 of
% Dr. Meyer's project guide.

if nargin < 1
   savefile = 'distG1C.mat';
end

if nargin < 2
   opt = struct('printFile', 1);
end

load(savefile); % load in the distance matrix 'dist'

% Load the list of songs
dataDir = getDir();
[wavList,genre] = textread([dataDir,'ground_truth.csv'],'%s %s','delimiter',',');
nSongs = length(wavList);
% Fix the names
wavList = strrep(wavList, '"', '');
wavList = strrep(wavList, 'mp3','wav');
genre   = strrep(genre, '"', '');

genreValues = getGenres(genre);

% Build confusion matrix
confMat = zeros(numel(unique(genreValues))); % 6x6
K = 5; % how many neighbors to find
for i=1:nSongs
   trueGenre = genreValues(i);

   neighbors = distMatknn(dist, i, K);
   neighborGenres = genreValues(neighbors);
   predGenre = mode(neighborGenres);
   confMat(predGenre, trueGenre) = confMat(predGenre, trueGenre) + 1;
end

% do some analysis on the confusion matrix
genreProb = zeros([numel(unique(genreValues)) 1]);
correctClassRate = zeros([numel(genreProb) 1]);
for i=1:numel(genreProb)
   correctClassRate(i) = confMat(i,i)/sum(genreValues == i);
   genreProb(i) = sum(genreValues == i)/nSongs;
end
%genreProb
%correctClassRate

probCorrect = sum(diag(confMat))/nSongs;
probCorrectNormalized = sum(correctClassRate*1/6);

fprintf(opt.printFile, 'Percent Correct: %3.2f%%\n', probCorrect*100);
fprintf(opt.printFile, 'Percent Correct (normalized): %3.2f%%\n', ...
   probCorrectNormalized*100);

end % confusionDistMat

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
