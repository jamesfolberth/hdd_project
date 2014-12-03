function [] = optimkNN(makePlot, plotFile)
% since deterministic, only need to take one sample
%featFile = 'featVecsWCH.mat';
featFile = 'featVecsDale.mat';

if nargin < 1
   makePlot = 0;
end

if nargin < 2 && makePlot ~= 0
   %plotFile = 'optimkNNPCA.mat';
   plotFile = 'optimkNNLLE.mat';
end

if ~makePlot

   method = 'pca';
   method = 'lle';

   switch method
   case 'pca'
      nVec = 5:3:50; % WCH
      %nVec = 5:5:80; % Dale
      nSamples = 5;
      probs = zeros([numel(nVec) nSamples]);
      for i=1:numel(nVec)
         for j=1:nSamples
            pcaOpt = struct('XValNum',10,'kNNNum',5,...
               'dimRed','pca','pcaNum',nVec(i));
            [~,~,probCorrect] = crossValkNNFeatVec(featFile, pcaOpt);
            %probCorrect = rand();
            probs(i,j) = probCorrect;
            fprintf(1, 'pcaNum = %d, probCorrect = %f\n', nVec(i), probCorrect);
         end
      end

      clear makePlot
      save('optimkNNPCA.mat');
      return

   case 'lle'
      kVec = 3:4:43;
      dimVec = 5:5:25;
      nSamples = 5;
      [K,D] = meshgrid(kVec,dimVec);
      probs = zeros([numel(K) 5]);
      for i = 1:numel(K)
         for j=1:nSamples
            kNNOpt = struct('XValNum', 10,'kNNNum',5,...
               'dimRed','lle','lleNum',K(i),'lleDim',D(i));
            [~,~,probCorrect] = crossValkNNFeatVec(featFile, kNNOpt);
            %probCorrect = rand();
            probs(i,j) = probCorrect;
            fprintf(1, 'k = %d, dim = %d, probCorrect = %f\n',...
               K(i), D(i), probCorrect);
         end
      end

      clear makePlot
      save('optimkNNLLE.mat');
      return
   
   otherwise
      error('bad method: %s', method);
   end

else
   load(plotFile, '-mat');

   switch method
   case 'pca'
      %plot(nVec, mean(probs),'o');
      errorbar(nVec, mean(probs,2), std(probs,0,2),'o')
      xlabel('PCA Num Terms'); ylabel('Classification Rate');
      title('kNN Clustering with PCs');
      print('Latex/figures/optimkNNPCA.pdf','-dpdf');
      
   case 'lle'
      %surf(K,D,reshape(mean(probs,2), size(K)));
      mu = mean(probs,2);
      sd = std(probs,0,2);
      %plot3(K(:), D(:), mu(:), 'o',...
      %      K(:), D(:), mu(:) + sd(:), 's',...
      %      K(:), D(:), mu(:) - sd(:), 's');
      hold on
      surf(K, D, reshape(mu, size(K)));
      for i=1:numel(K)
         plot3([K(i);K(i)], [D(i);D(i)], [mu(i)-sd(i); mu(i)+sd(i)],...
            '-k','LineWidth',2);
      end
      hold off
      xlabel('lleNum'); ylabel('lleDim'); %zlabel('Classification Rate');

      [~,ind] = max(mu);
      fprintf(1,'max mean(probCorrect) = %f at lleNum = %d, lleDim = %d\n',...
         probs(ind), K(ind), D(ind));

      % make 1d plot
      close
      figure
      mu = reshape(mu, size(K));
      sd  = reshape(sd, size(K));
      errorbar(K(1,:), mu(1,:), sd(1,:),'o');
      %errorbar(K(2,:), mu(2,:), sd(2,:),'og');
      xlabel('LLE Number of Neighbors');
      ylabel('Classification Rate');
      title(sprintf('kNN Clustering after LLE into %d dims',D(1,1)));
      print('Latex/figures/optimkNNLLE.pdf','-dpdf');
   otherwise
      error('bad method: %s', method);
   end

end
