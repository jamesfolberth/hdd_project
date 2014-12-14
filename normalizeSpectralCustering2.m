function U = normalizeSpectralCustering2(k,G)


nG=(full(diag(diag(G)))^(-.5)*G*full(diag(diag(G)))^(-.5));
[V,D] = eig(G,diag(diag(G))); 
[m,index] = sort(abs(diag(D)),'descend');
V = V(:,index(1:k)); 
U = zeros(size(G,1),k);
for i = 1:size(U,1)
    for j = 1:size(U,2)
        U(i,j) = V(i,j)/ (sum(V(i,:)*V(i,:)')^(.5));
    end
end

