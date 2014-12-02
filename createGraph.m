function G = createGraph(opt)
%Creates Unnormalized Laplaicain 
%opt =1 epsilon graph; opt =2 k-nearest neighbor graphs; opt = 3

opt = 1;
savefile = 'distG1C.mat';
 

distMat = load(savefile,'dist');
 distMat = distMat.dist;
%  distMat = distMat - distMat(1,1); 
dataDir = getDir();

m = min(min(distMat));
M = max(max(distMat));

if opt ==1
    epsilon  = mean(mean(distMat)); 
    S = zeros(size(distMat)); 
    S(find(distMat<epsilon)) = distMat(find(distMat<=epsilon)); 
    D = diag(sum(S,1)); 
    G = D-S; 
    
elseif opt ==2
    K = 5
    for i = 1:size(distMat,1)
            
        neighbors = distMatknn(distMat, i, K);
        S = zeros(size(distMat)); 
        S(i,neighbors) = distMat(i,neighbors); 
    end
    S  = S+S'; 
    D = diag(sum(S,1)); 
    G = D-S; 
    

    
elseif opt ==3
    S = distMat; 
    D = diag(sum(S,1)); 
    G = D-S; 
else
    disp('Incorrect input')
    return
    
end