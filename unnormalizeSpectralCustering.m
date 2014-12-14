function V = unnormalizeSpectralCustering(k,G)

[V,D] = eig(G); 
[m,index] = sort(abs(diag(D)),'descend');
V = V(:,index(1:k)); 
