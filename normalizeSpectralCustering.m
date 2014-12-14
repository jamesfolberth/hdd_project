function V = normalizeSpectralCustering(k,G)

[V,D] = eig(G,diag(diag(G))); 
[m,index] = sort(abs(diag(D)),'descend');
V = V(:,index(1:k)); 

