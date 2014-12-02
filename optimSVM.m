function [] = optimSVM(makePlot, plotFile)
% since deterministic, only need to take one sample
featFile = 'featVecsWCH.mat';
%featFile = 'featVecsDale.mat';

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

   switch method
   case 'onevall - order'
      order = 2.0:.25:2.75;
      nSamples = 5;
      probs = zeros([numel(order) nSamples]);
      for i=1:numel(order)
         for j=1:nSamples
            SVMOpt = struct('XValNum',10,'dimRed','none',...
               'MCMethod','onevall','SVMOrder',order(i));
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
            SVMOpt = struct('XValNum',10,'dimRed','none',...
               'MCMethod','onevone','SVMOrder',order(i));
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
            SVMOpt = struct('XValNum',10,'dimRed','none',...
               'MCMethod','ECOC','SVMOrder',order(i));
            [~,~,probCorrect] = crossValSVMFeatVec(featFile, SVMOpt);
            %probCorrect = rand();
            probs(i,j) = probCorrect;
            fprintf(1, 'SVMOrder = %f, probCorrect = %f\n', order(i), probCorrect);
         end
      end

      clear makePlot
      save('optimSVMECOCOrder.mat');
      return

   otherwise
      error('bad method: %s', method);
   end

else
   load(plotFile, '-mat');

   switch method
   case 'onevall - order'
      errorbar(order, mean(probs,2), std(probs,0,2),'o')
      xlabel('SVM Poly Kernel Order'); ylabel('Classification Rate');
      title('Classification Rate - One vs. All');

   case 'onevone - order'
      errorbar(order, mean(probs,2), std(probs,0,2),'o')
      xlabel('SVM Poly Kernel Order'); ylabel('Classification Rate');
      title('Classification Rate - One vs. One');

   case 'ECOC - order'
      errorbar(order, mean(probs,2), std(probs,0,2),'o')
      xlabel('SVM Poly Kernel Order'); ylabel('Classification Rate');
      title('Classification Rate - Error Correcting');
 
   otherwise
      error('bad method: %s', method);
   end

end
