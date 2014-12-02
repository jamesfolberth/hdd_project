function [] = testPCA()

load featVecsWCH.mat;

% standardize data
data = transpose(feat); % data along rows is convention
mu = mean(data, 1);
data = bsxfun(@minus, data, mu);
%variance = var(data, 0, 1);
%data = bsxfun(@times, data, 1./sqrt(variance));

[U, S, V] = svd(data, 'econ');

latent = diag(S).^2 / (size(data,1)-1);
explained = 100*latent/sum(latent);
plot(cumsum(explained), 'o')

%disp([(1:numel(explained))' cumsum(explained)]);

end
