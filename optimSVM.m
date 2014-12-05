function [] = optimSVM(makePlot, plotFile)
%featFile = 'featVecsWCH.mat'; featType = 'WCH';
featFile = 'featVecsDale.mat'; featType = 'Dale';

if nargin < 1
   makePlot = 0;
end

if nargin < 2 && makePlot ~= 0
   plotFile = 'optimSVM.mat';
end

if ~makePlot

   method = 'onevall - order';
   method = 'onevone - order';
   method = 'ECOC - order';

   MCMethod = 'onevall'; % XXX used only for prDim
   prMode = 'genre0.5';
   method = 'prDim';

   switch method
   % {{{ * - order
   case 'onevall - order'
      order = 2.0:.25:2.75;
      nSamples = 5;
      probs = zeros([numel(order) nSamples]);
      for i=1:numel(order)
         for j=1:nSamples
            switch featType
            case 'WCH'
               SVMOpt = struct('XValNum',10,'dimRed','none',...
                  'MCMethod','onevall','SVMOrder',order(i));
            case 'Dale'
               SVMOpt = struct('XValNum',10,'dimRed','pr','prDim',67,...
                  'MCMethod','onevall','SVMOrder',order(i));
            otherwise
               error('Bad feature type: %s.  Specify with featFile', featType);
            end
           
            [~,~,probCorrect] = crossValSVMFeatVec(featFile, SVMOpt);
            %probCorrect = rand();
            probs(i,j) = probCorrect;
            fprintf(1, 'SVMOrder = %f, probCorrect = %f\n', order(i), probCorrect);
         end
      end

      clear makePlot
      save('optimSVMOVAOrder.mat');
      return

   case 'onevone - order'
      order = 2.0:.25:2.75;
      nSamples = 5;
      probs = zeros([numel(order) nSamples]);
      for i=1:numel(order)
         for j=1:nSamples
            switch featType
            case 'WCH'
               SVMOpt = struct('XValNum',10,'dimRed','none',...
                  'MCMethod','onevone','SVMOrder',order(i));
            case 'Dale'
               SVMOpt = struct('XValNum',10,'dimRed','pr','prDim',67,...
                  'MCMethod','onevone','SVMOrder',order(i));
            otherwise
               error('Bad feature type: %s.  Specify with featFile', featType);
            end
            
            [~,~,probCorrect] = crossValSVMFeatVec(featFile, SVMOpt);
            %probCorrect = rand();
            probs(i,j) = probCorrect;
            fprintf(1, 'SVMOrder = %f, probCorrect = %f\n', order(i), probCorrect);
         end
      end

      clear makePlot
      save('optimSVMOVOOrder.mat');
      return

   case 'ECOC - order'
      order = 2.0:.25:2.75;
      nSamples = 5;
      probs = zeros([numel(order) nSamples]);
      for i=1:numel(order)
         for j=1:nSamples
            switch featType
            case 'WCH'
               SVMOpt = struct('XValNum',10,'dimRed','none',...
                  'MCMethod','ECOC','SVMOrder',order(i));
            case 'Dale'
               SVMOpt = struct('XValNum',10,'dimRed','pr','prDim',67,...
                  'MCMethod','ECOC','SVMOrder',order(i));
            otherwise
               error('Bad feature type: %s.  Specify with featFile', featType);
            end

            [~,~,probCorrect] = crossValSVMFeatVec(featFile, SVMOpt);
            %probCorrect = rand();
            probs(i,j) = probCorrect;
            fprintf(1, 'SVMOrder = %f, probCorrect = %f\n', order(i), probCorrect);
         end
      end

      clear makePlot
      save('optimSVMECOCOrder.mat');
      return

   % }}}

   case 'prDim'
      dim = 20:5:150;
      nSamples = 5;
      probs = zeros([numel(dim) nSamples]);
      genreClassRate = zeros([6 numel(dim) nSamples]);
      for i=1:numel(dim)
         for j=1:nSamples
            switch featType
            case 'WCH'
               SVMOpt = struct('XValNum',10,'dimRed','pr','prMode',prMode,...
                  'prDim',dim(i),'MCMethod',MCMethod,'SVMOrder',2.5);
            case 'Dale'
               SVMOpt = struct('XValNum',10,'dimRed','pr','prMode',prMode,...
                  'prDim',dim(i),'MCMethod',MCMethod,'SVMOrder',2.5);
            otherwise
               error('Bad feature type: %s.  Specify along with featFile', featType);
            end
           
            [confAvg,confSD,probCorrect]=crossValSVMFeatVec(featFile, SVMOpt);
            %probCorrect = rand();

            genreClassRate(:,i,j) = diag(confAvg)... 
               ./reshape(sum(confAvg,1), [6 1]);
            probs(i,j) = probCorrect;
            fprintf(1, 'prDim = %d, probCorrect = %f\n', dim(i), probCorrect);
         end
      end

      clear makePlot
      save('optimSVMOVAprDim.mat');
      return

   otherwise
      error('bad method: %s', method);
   end

else
   load(plotFile, '-mat');

   switch method
   case 'onevall - order'
      errorbar(order, mean(probs,2), std(probs,0,2),'o')
      axis([1.9 2.85 0.6 0.75]);
      xlabel('SVM Poly Kernel Order'); ylabel('Classification Rate');
      title('Classification Rate - One vs. All');
      %print('Latex/figures/optimSVMOVAOrder_WCH.pdf','-dpdf')
      print('Latex/figures/optimSVMOVAOrder_Dale.pdf','-dpdf')

   case 'onevone - order'
      errorbar(order, mean(probs,2), std(probs,0,2),'o')
      axis([1.9 2.85 0.6 0.75]);
      xlabel('SVM Poly Kernel Order'); ylabel('Classification Rate');
      title('Classification Rate - One vs. One');
      %print('Latex/figures/optimSVMOVOOrder_WCH.pdf','-dpdf')
      print('Latex/figures/optimSVMOVOOrder_Dale.pdf','-dpdf')

   case 'ECOC - order'
      errorbar(order, mean(probs,2), std(probs,0,2),'o')
      axis([1.9 2.85 0.6 0.75]);
      xlabel('SVM Poly Kernel Order'); ylabel('Classification Rate');
      title('Classification Rate - Error Correcting');
      %print('Latex/figures/optimSVMECOCOrder_WCH.pdf','-dpdf')
      print('Latex/figures/optimSVMECOCOrder_Dale.pdf','-dpdf')

   case {'prDim','onevall-prDim'}
      errorbar(dim, mean(probs,2), std(probs,0,2), 'o');
      xlabel('Dimension'); ylabel('Classification Rate');
      title('Classification Rate - One vs. All');
      %print('Latex/figures/optimSVMOVAprDim_Dale.pdf','-dpdf');
      print('Latex/figures/optimSVMOVAprDim_genre05_Dale.pdf','-dpdf');

      %errorbar(repmat(transpose(dim), [1 6]), transpose(mean(genreClassRate,3)),...
      %   transpose(std(genreClassRate,0,3)));
      %axis([min(dim)-1 max(dim)+1 0.4 1])
      %xlabel('Dimension'); ylabel('Classification Rate');
      %title('Classification Rate - One vs. All');
      %legend('c','e','j\_b','m\_p','r\_p','w','Orientation','horizonta',...
      %   'Location','SouthEast');
      %print('Latex/figures/optimSVMOVAprDimClass_Dale.pdf','-dpdf');


 
   otherwise
      error('bad method: %s', method);
   end

end
