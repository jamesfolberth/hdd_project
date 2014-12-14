function [G1,G2,G3] = createGraph(sigma)
%Creates Unnormalized Laplaicain 
%opt =1 epsilon graph; opt =2 k-nearest neighbor graphs; opt = 3

savefile = 'featVecsDale.mat';
 

distMat = load(savefile);
  distMat = distMat.feat;
    distMat = normr(distMat);
%  distMat = distMat - distMat(1,1); 
% dataDir = getDir();
% 
% m = min(min(distMat));
% M = max(max(distMat));

Similarty = zeros(size(distMat,2),size(distMat,2));
for i =1:size(distMat,2)
    for j = 1:size(distMat,2)
        Similarty(i,j) = exp(-norm(distMat(:,i) - distMat(:,j),2)^2/(2*sigma^2));
    end
end




     epsilon  = mean(mean(Similarty)); 
      espilon = .04;
    S = zeros(size(Similarty)); 
    S(find(Similarty>epsilon)) = Similarty(find(Similarty>epsilon)); 
    D = diag(sum(S,1)); 
    G1 = D-S; 
    

    k = 5;
     S = zeros(size(Similarty));
    for i = 1:size(Similarty,1)
            
        neighbors =distMatknn(Similarty, i,k);
        
        S(i,neighbors) = Similarty(i,neighbors); 
    end
    S  = S+S'; 
    D = diag(sum(S,1)); 
    G2 = D-S; 
    

    

    S = Similarty; 
    D = diag(sum(S,1)); 
    G3 = D-S; 


