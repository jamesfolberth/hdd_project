function IDX = normalizeSpectralCustering(k,opt)

G = createGraph(opt); 
[V,D] = eig(G,diag(diag(G))); 
[m,index] = sort(abs(diag(D)),'descend');
V = V(:,index(1:k)); 
IDX  = kmeans(V, 6);
