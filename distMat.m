function [dist] = distMat(wavList, opt)
% combine various measures to produce a single distance matrix
% Input:
%    wavList - list of tracks (with correct path to file)
%    opt     - options structure
%
% Output:
%    dist - matrix of distances

% populate wavList with default tracks
if nargin < 1
   dataDir = getDir();

   % Load the list of songs
   [wavList,genre] = textread([dataDir,'ground_truth.csv'],'%s %s','delimiter',',');
   % Fix the names
   wavList = strrep( wavList, '"', '');
   wavList = strrep( wavList, 'mp3','wav');
end

% populate options structure with default method, etc.
if nargin < 2
   opt = struct('method','G1C','savefile','distG1C.mat');
end

% Set up variables common to all methods
nSongs = length(wavList);
%nSongs = 5; % for testing

printFile = 1; % stdout
%printFile = fopen('/dev/null');

switch opt.method
case 'G1C'

   % NOTE: melCoeffs will use about 1GB with DCT compression
   mfccOpts = struct('segLength',512',...
                     'shiftLength',256',...
                     'method','dct',...
                     'numTerms',20);
   melCoeffs = cell(nSongs,1);

   % Compute MFCCs
   for(i = 1:nSongs)
      fprintf(printFile,'\rMFCC: %d of %d.',i, nSongs);
      wavFile = strcat(dataDir, wavList{i});
      % wavread will be deprecated, so use audioread
      if ( ~exist('audioread') )
         [wav,fs] = wavread(wavFile);
      else
         [wav,fs] = audioread(wavFile,'double');
      end

      wav = wav*10^(96/20);

      melCoeffs{i} = mfcc(wav,fs,mfccOpts);
   end
   fprintf(printFile, '\n');
   
   % Compute various distance matrices and associated mean, std. dev.
   G1Opt = struct();
   G1Mat = G1DistMat(melCoeffs, G1Opt);
   
   G1Mean = mean(G1Mat(:));
   G1Std  = std(G1Mat(:));

   FPDistMatsOpt = struct('method',opt.method);
   FPMats = FPDistMats(melCoeffs, mfccOpts, FPDistMatsOpt);

   FPMean = mean(FPMats.FP(:));
   FPStd  = std(FPMats.FP(:));

   FPGMean = mean(FPMats.FPG(:));
   FPGStd  = std(FPMats.FPG(:));

   FPBassMean = mean(FPMats.FPBass(:));
   FPBassStd  = std(FPMats.FPBass(:));

   % Combine distance matrices with normalized weighted sum; Eqn (2.42) in Pamp
   dist = 0.7*(G1Mat - G1Mean)/G1Std + ...
          0.1*(FPMats.FP - FPMean)/FPStd + ...
          0.1*(FPMats.FPG - FPGMean)/FPGStd + ...
          0.1*(FPMats.FPBass - FPBassMean)/FPBassStd + ...
          10; % sufficient to ensure all values are greater than 1
              % see Pampalk's ma_g1c_ComputeSimilarities.m

   %fprintf(printFile, 'Number of negative values in dist: %d of %d\n',...
   %   nnz(dist<0), numel(dist));
   
   save(opt.savefile,'dist');

otherwise
   error(sprintf('Unknown method to combine metrics: opt.method = %s',...
   opt.method));
end


% if not stdout or stderr, close the file
if printFile > 2
   fclose(printFile);
end

end % distMat
