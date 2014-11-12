function [knnIndexes] = distMatknn(dist, songIndex, k)
% Given a matrix if distances between songs and a song index, return the 
% indexes of the k nearest neighbors of the song with index songIndex 
% according to the distances in dist

% rank the other songs based on their distances
if songIndex == 1
   [rank, index] = sort(dist(2:end, songIndex), 1, 'ascend');
elseif songIndex == size(dist,1)
   [rank, index] = sort(dist(1:end-1, songIndex), 1, 'ascend');
else
   [rank, index] = sort([dist(1:songIndex-1,songIndex); ...
                         dist(songIndex+1:end,songIndex)], 1, 'ascend');
end

knnIndexes = index(1:k);

end 
