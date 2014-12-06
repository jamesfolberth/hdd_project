function [] = optimkNN(makePlot, plotFile)
%featFile = 'featVecsWCH.mat'; featType = 'WCH';
%featFile = 'featVecsDale.mat'; featType = 'Dale';
featFile = 'featVecsAppend.mat'; featType = 'Dale';
%featFile = 'distG1C.mat'; featType = 'dist';



if nargin < 1
   makePlot = 0;
end

if nargin < 2 && makePlot ~= 0
   %plotFile = 'optimkNNPCA.mat';
   plotFile = 'optimkNNLLE.mat';
end

if ~makePlot

   %method = 'pca';
   %method = 'lle';
   %method = 'lda';

   method = 'prDim';
   prMode = 'genre0.5';

   switch method
   case 'pca'
      %nVec = 5:3:50; % WCH
      %nVec = 5:5:80; % Dale
      nVec = 5:5:150; % dist
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
  
   case 'prDim'
      dim = 20:5:150;
      nSamples = 5;
      probs = zeros([numel(dim) nSamples]);
      genreClassRate = zeros([6 numel(dim) nSamples]);
      for i=1:numel(dim)
         for j=1:nSamples
            switch featType
            case 'WCH'
               kNNOpt = struct('XValNum',10,'dimRed','pr','prMode',prMode,...
                  'prDim',dim(i));
            case 'Dale'
               kNNOpt = struct('XValNum',10,'dimRed','pr','prMode',prMode,...
                  'prDim',dim(i));
            case 'dist'
               kNNOpt = struct('XValNum',10,'dimRed','pr','prMode',prMode,...
                  'prDim',dim(i));
            otherwise
               error('Bad feature type: %s.  Specify along with featFile',...
                  featType);
            end
           
            [confAvg,confSD,probCorrect]=crossValkNNFeatVec(featFile, kNNOpt);
            %probCorrect = rand(); confAvg = rand([6 6]);

            genreClassRate(:,i,j) = diag(confAvg)... 
               ./reshape(sum(confAvg,1), [6 1]);
            probs(i,j) = probCorrect;
            fprintf(1, 'prDim = %d, probCorrect = %f\n', dim(i), probCorrect);
         end
      end

      clear makePlot
      save('optimkNNprDim.mat');
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

   case 'prDim'
      errorbar(dim, mean(probs,2), std(probs,0,2), 'o');
      xlabel('Dimension'); ylabel('Classification Rate');
      title('Classification Rate - kNN, k=5');
      preparePrint()
      print('Latex/figures/optimkNNprDim_genre0.5_swt.pdf','-dpdf');

      %errorbar(repmat(transpose(dim), [1 6]), transpose(mean(genreClassRate,3)),...
      %   transpose(std(genreClassRate,0,3)));
      %%axis([min(dim)-1 max(dim)+1 0.4 1])
      %xlabel('Dimension'); ylabel('Classification Rate');
      %title('Classification Rate - One vs. All');
      %legend('c','e','j\_b','m\_p','r\_p','w','Orientation','horizonta',...
      %   'Location','SouthEast');
      %print('Latex/figures/optimSVMOVAprDimClass_Dale.pdf','-dpdf');




   otherwise
      error('bad method: %s', method);
   end

end

end % optimkNN

function [] = preparePrint()

% Stolen from http://tipstrickshowtos.blogspot.com/2010/08/how-to-get-rid-of-white-margin-in.html
ti = get(gca,'TightInset');
set(gca,'Position',[ti(1) ti(2) 1-ti(3)-ti(1) 1-ti(4)-ti(2)]);

set(gca,'units','centimeters')
pos = get(gca,'Position');
ti = get(gca,'TightInset');

set(gcf, 'PaperUnits','centimeters');
set(gcf, 'PaperSize', [pos(3)+ti(1)+ti(3) pos(4)+ti(2)+ti(4)]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition',[0 0 pos(3)+ti(1)+ti(3) pos(4)+ti(2)+ti(4)]);

end
