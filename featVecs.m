function [feat] = featVecs(wavList, opt)
% combine various features to create a matrix of feature vectors
% Input:
%    wavList - list of tracks (with correct path to file)
%    opt     - options structure
%
% Output:
%    feat    - feature length x number of songs matrix

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
   opt = struct('method','WCH','savefile','featVecsWCH.mat');
   %opt = struct('method','Dale','savefile','featVecsDale.mat');
end

% Set up variables common to all methods
nSongs = length(wavList);
%nSongs = 10; % for testing

printFile = 1; % stdout
%printFile = fopen('/dev/null');

switch opt.method
case 'WCH'

   %mfccOpts = struct('segLength',1024,'shiftLength',1024,...
   mfccOpts = struct('segLength',512,'shiftLength',256,...
                     'method','dct','numTerms',20);
   textureOpts = struct('segLength',11024,'shiftLength',11024/2,...
                         'method','dct','numTerms',6);
   wchOpts = struct('wName','bior4.4','nLevels',7,'segLength',2^18);
   

   % Compute features
   feat = zeros([66 nSongs]);
   % 01:10   - assorted simple features
   % 11:46   - mean and var of MFCC DCT coeffs 2:19
   % 47:66   - wavelet coeff histogram features (1st 3 moments + energy)

   for(i = 1:nSongs)
      fprintf(printFile,'\rSong: %d of %d.',i, nSongs);
      wavFile = strcat(dataDir, wavList{i});
      % wavread will be deprecated, so use audioread
      if ( ~exist('audioread') )
         [wav,fs] = wavread(wavFile);
      else
         [wav,fs] = audioread(wavFile,'double');
      end

      wav = wav*10^(96/20);

      % only store data for the current song
      [melCoeffs, melPowDB, pow] = mfcc(wav, fs, mfccOpts);
      fpMed = flucPat(melPowDB);
      sfeat = simpFeat(wav, fs, pow, melPowDB, fpMed);

      feat(01,i) = sfeat.zcr;

      % Spectral centroid, rolloff, and flux stolen from Dale
      specC = (1:36)*melPowDB./sum(melPowDB,1);
      specR = zeros(size(melPowDB,2),1);
      for t = 1:size(melPowDB,2)
         inds = find(cumsum(melPowDB(:,t)) >= 0.85*sum(melPowDB(:,t)),1);
         specR(t) = inds;
      end
      normMS = bsxfun(@times, melPowDB, 1./sum(melPowDB,1));
      specF = sum( (normMS(:,2:length(melPowDB)) ...
      - normMS(:,1:length(melPowDB)-1)).^2,1);
      feat(2:7,i) = [mean(specC); var(specC);
                   mean(specR); var(specR);
                   mean(specF); var(specF)];

      feat(08,i) = sfeat.fpMax;
      feat(09,i) = sfeat.fpSum;
      feat(10,i) = sfeat.fpBass; 
      feat(11,i) = sfeat.fpAggr;
      feat(12,i) = sfeat.fpDLF;
      feat(13,i) = sfeat.fpG;
      feat(14,i) = sfeat.fpF;

      feat(15:32,i) = mean(melCoeffs(2:19,:),2);
      feat(33:50,i) = var(melCoeffs(2:19,:),0,2);
      
      %melDCT = mfcc(wav, fs, textureOpts);
      %melDCT = melDCT(2:6,:);
      %feat(15:24,i) = [mean(melDCT,2); var(melDCT,0,2)];

      %feat(25:40,i) = wch(wav);
      %feat(51:66,i) = wch(wav);
      [aFeat,dFeat] = wch(wav, wchOpts);
      feat(51:58,i) = aFeat(end-7:end); % use the features from the
      feat(59:66,i) = dFeat(end-7:end); % last two A and D sub-bands

   end
   fprintf(printFile, '\n');

   save(opt.savefile,'feat');

case 'Dale'
    addpath('aca_tlbx')
    %analysisOpts = struct('segLength',1024,'shiftLength',1024,...
    %                      'method','raw');
    %textureOpts = struct('segLength',11024,'shiftLength',11024/2,...
    %                      'method','dct','numTerms',6);
    wchOpts = struct('wName','bior4.4','nLevels',7,'segLength',2^18);

    % Compute features
    feat = [];

    spectralFeatures = {'SpectralCentroid', ...     %1:4
                        'SpectralCrest', ...        %5:8
                        'SpectralDecrease', ...     %9:12
                        'SpectralFlatness', ...     %13:16
                        'SpectralFlux', ...         %17:20
                        'SpectralKurtosis', ...     %21:24
                        'SpectralMfccs', ...        %25:76
                        'SpectralPitchChroma', ...  %77:124
                        'SpectralRolloff', ...      %125:128
                        'SpectralSkewness', ...     %129:132
                        'SpectralSlope', ...        %133:136
                        'SpectralSpread', ...       %137:140
                        'SpectralTonalPowerRatio'}; %141:144

    temporalFeatures = {'TimeAcfCoeff', ...         %145:148
                        'TimeMaxAcf', ...           %149:152
                        'TimePeakEnvelope', ...     %153:160
                        'TimePredictivityRatio', ...%161:164
                        'TimeRms', ...              %165:168
                        'TimeStd', ...              %169:172
                        'TimeZeroCrossingRate'};    %173:176
                    
                        %Fluctation Patterns        %177:182
                        
                        %WCH                        %183:198
                
    iHopLength = 1024;
    iBlockLength = 2048;
    afWindow = hann(iBlockLength,'periodic');
   
    feat = [];
    
    for(i = 1:nSongs)
      featV = [];
      
      fprintf(printFile,'\rSong: %d of %d.',i, nSongs);
      wavFile = strcat(dataDir, wavList{i});
      % wavread will be deprecated, so use audioread
      if ( ~exist('audioread') )
         [wav,fs] = wavread(wavFile);
      else
         [wav,fs] = audioread(wavFile,'double');
      end

      wav = wav*10^(96/20);
      wav = wav(wav~=0);

      % Grab middle 200 seconds
      if(length(wav) > 2205002)
          wav = wav((floor(length(wav)/2)-1102500):(floor(length(wav)/2)+1102501));
      end
      
      %N = length(wav);

      % Compute the spectrum
      [X,~,~] = spectrogram(  wav,...
                              afWindow,...
                              iBlockLength-iHopLength,...
                              iBlockLength,...
                              fs);

      % Compute the magnitude spectrum
      Xmag = abs(X)*2/iBlockLength;
      
      % Compute Spectral Features
      for( k = 1:length(spectralFeatures) )
          hFeatureFunc = str2func(['Feature' spectralFeatures{k}]);

          v = hFeatureFunc(Xmag, fs);

          featV = [featV; mean(v,2); var(v,0,2); ...
                  skewness(v,1,2); kurtosis(v,1,2)];
      end
      % Compute Temporal Features
      
      for( k = 1:length(temporalFeatures) )
          hFeatureFunc = str2func(['Feature' temporalFeatures{k}]);

          [v,~] = hFeatureFunc(wav, iBlockLength, iHopLength, fs);

          featV = [featV; mean(v,2); var(v,0,2); ...
                      skewness(v,1,2); kurtosis(v,1,2)];
      end
      
      melS = pow2Mel(X,fs,struct('segLength',iBlockLength,'shiftLength',iHopLength));
      
      % Fluctuation patterns
      fpMed = flucPat(melS);
      fp = reshape(fpMed, [12 30]);
      fpMax = max(fp(:)); %177
      fpBass = sum(sum(fp(1:2,3:end))); %178
      fpAggr = sum(sum(fp(2:end,1:4)))/fpMax; %179
      fpDLF = sum(sum(fp(1:3,:)))/sum(sum(fp(9:end,:))); %180
      fpGrav = sum( fp*(1:size(fp,2))' )/sum(fp(:)); %181
      fpFoc = mean(fp(:))/fpMax; %182

      featV = [featV; fpMax; fpBass; fpAggr; fpDLF; fpGrav; fpFoc];
      
      [aFeat,dFeat] = wch(wav, wchOpts);
      %featV = [featV; aFeat; dFeat];
      featV = [featV; aFeat(end-7:end); dFeat(end-7:end)];
 
      feat = [feat, featV];
    end
    
    fprintf(printFile, '\n');

    save(opt.savefile,'feat');

case 'testWCH'

   %mfccOpts = struct('segLength',1024,'shiftLength',1024,...
   mfccOpts = struct('segLength',512,'shiftLength',256,...
                     'method','dct','numTerms',20);
   textureOpts = struct('segLength',11024,'shiftLength',11024/2,...
                         'method','dct','numTerms',6);
   wchOpts = struct('wName','bior4.4','nLevels',7,'segLength',2^18);

   % Compute features
   feat = zeros([66 nSongs]);
   % 01:10   - assorted simple features
   % 11:46   - mean and var of MFCC DCT coeffs 2:19
   % 47:66   - wavelet coeff histogram features (1st 3 moments + energy)

   for(i = 1:nSongs)
      fprintf(printFile,'\rSong: %d of %d.',i, nSongs);
      wavFile = wavList{i};
      % wavread will be deprecated, so use audioread
      if ( ~exist('audioread') )
         [wav,fs] = wavread(wavFile);
      else
         [wav,fs] = audioread(wavFile,'double');
      end

      % downsample to 11025 Hz
      fs = fs/2;
      wav = wav(1:2:end);

      wav = wav*10^(96/20);

      % only store data for the current song
      [melCoeffs, melPowDB, pow] = mfcc(wav, fs, mfccOpts);
      fpMed = flucPat(melPowDB);
      sfeat = simpFeat(wav, fs, pow, melPowDB, fpMed);

      feat(01,i) = sfeat.zcr;

      % Spectral centroid, rolloff, and flux stolen from Dale
      specC = (1:36)*melPowDB./sum(melPowDB,1);
      specR = zeros(size(melPowDB,2),1);
      for t = 1:size(melPowDB,2)
         inds = find(cumsum(melPowDB(:,t)) >= 0.85*sum(melPowDB(:,t)),1);
         specR(t) = inds;
      end
      normMS = bsxfun(@times, melPowDB, 1./sum(melPowDB,1));
      specF = sum( (normMS(:,2:length(melPowDB)) ...
      - normMS(:,1:length(melPowDB)-1)).^2,1);
      feat(2:7,i) = [mean(specC); var(specC);
                   mean(specR); var(specR);
                   mean(specF); var(specF)];

      feat(08,i) = sfeat.fpMax;
      feat(09,i) = sfeat.fpSum;
      feat(10,i) = sfeat.fpBass; 
      feat(11,i) = sfeat.fpAggr;
      feat(12,i) = sfeat.fpDLF;
      feat(13,i) = sfeat.fpG;
      feat(14,i) = sfeat.fpF;

      feat(15:32,i) = mean(melCoeffs(2:19,:),2);
      feat(33:50,i) = var(melCoeffs(2:19,:),0,2);
      
      %melDCT = mfcc(wav, fs, textureOpts);
      %melDCT = melDCT(2:6,:);
      %feat(15:24,i) = [mean(melDCT,2); var(melDCT,0,2)];

      %feat(25:40,i) = wch(wav);
      %feat(51:66,i) = wch(wav);
      [aFeat,dFeat] = wch(wav, wchOpts);
      feat(51:58,i) = aFeat(end-7:end); % use the features from the
      feat(59:66,i) = dFeat(end-7:end); % last two A and D sub-bands

   end
   fprintf(printFile, '\n');

   save(opt.savefile,'feat');


case 'testDale'
    addpath('aca_tlbx')
    %analysisOpts = struct('segLength',1024,'shiftLength',1024,...
    %                      'method','raw');
    %textureOpts = struct('segLength',11024,'shiftLength',11024/2,...
    %                      'method','dct','numTerms',6);
    wchOpts = struct('wName','bior4.4','nLevels',7,'segLength',2^18);

    % Compute features
    feat = [];

    spectralFeatures = {'SpectralCentroid', ...     %1:4
                        'SpectralCrest', ...        %5:8
                        'SpectralDecrease', ...     %9:12
                        'SpectralFlatness', ...     %13:16
                        'SpectralFlux', ...         %17:20
                        'SpectralKurtosis', ...     %21:24
                        'SpectralMfccs', ...        %25:76
                        'SpectralPitchChroma', ...  %77:124
                        'SpectralRolloff', ...      %125:128
                        'SpectralSkewness', ...     %129:132
                        'SpectralSlope', ...        %133:136
                        'SpectralSpread', ...       %137:140
                        'SpectralTonalPowerRatio'}; %141:144

    temporalFeatures = {'TimeAcfCoeff', ...         %145:148
                        'TimeMaxAcf', ...           %149:152
                        'TimePeakEnvelope', ...     %153:160
                        'TimePredictivityRatio', ...%161:164
                        'TimeRms', ...              %165:168
                        'TimeStd', ...              %169:172
                        'TimeZeroCrossingRate'};    %173:176
                    
                        %Fluctation Patterns        %177:182
                        
                        %WCH                        %183:198
                
    iHopLength = 1024;
    iBlockLength = 2048;
    afWindow = hann(iBlockLength,'periodic');
   
    feat = [];
    
    for(i = 1:nSongs)
      featV = [];
      
      fprintf(printFile,'\rSong: %d of %d.',i, nSongs);
      wavFile = wavList{i};
      % wavread will be deprecated, so use audioread
      if ( ~exist('audioread') )
         [wav,fs] = wavread(wavFile);
      else
         [wav,fs] = audioread(wavFile,'double');
      end

      % downsample to 11025 Hz
      fs = fs/2;
      wav = wav(1:2:end);

      wav = wav*10^(96/20);
      wav = wav(wav~=0);

      % Grab middle 200 seconds
      if(length(wav) > 2205002)
          wav = wav((floor(length(wav)/2)-1102500):(floor(length(wav)/2)+1102501));
      end
      
      %N = length(wav);

      % Compute the spectrum
      [X,~,~] = spectrogram(  wav,...
                              afWindow,...
                              iBlockLength-iHopLength,...
                              iBlockLength,...
                              fs);

      % Compute the magnitude spectrum
      Xmag = abs(X)*2/iBlockLength;
      
      % Compute Spectral Features
      for( k = 1:length(spectralFeatures) )
          hFeatureFunc = str2func(['Feature' spectralFeatures{k}]);

          v = hFeatureFunc(Xmag, fs);

          featV = [featV; mean(v,2); var(v,0,2); ...
                  skewness(v,1,2); kurtosis(v,1,2)];
      end
      % Compute Temporal Features
      
      for( k = 1:length(temporalFeatures) )
          hFeatureFunc = str2func(['Feature' temporalFeatures{k}]);

          [v,~] = hFeatureFunc(wav, iBlockLength, iHopLength, fs);

          featV = [featV; mean(v,2); var(v,0,2); ...
                      skewness(v,1,2); kurtosis(v,1,2)];
      end
      
      melS = pow2Mel(X,fs,struct('segLength',iBlockLength,'shiftLength',iHopLength));
      
      % Fluctuation patterns
      fpMed = flucPat(melS);
      fp = reshape(fpMed, [12 30]);
      fpMax = max(fp(:)); %177
      fpBass = sum(sum(fp(1:2,3:end))); %178
      fpAggr = sum(sum(fp(2:end,1:4)))/fpMax; %179
      fpDLF = sum(sum(fp(1:3,:)))/sum(sum(fp(9:end,:))); %180
      fpGrav = sum( fp*(1:size(fp,2))' )/sum(fp(:)); %181
      fpFoc = mean(fp(:))/fpMax; %182

      featV = [featV; fpMax; fpBass; fpAggr; fpDLF; fpGrav; fpFoc];
      
      [aFeat,dFeat] = wch(wav, wchOpts);
      %featV = [featV; aFeat; dFeat];
      featV = [featV; aFeat(end-7:end); dFeat(end-7:end)];
      
      feat = [feat, featV];
    end
    
    fprintf(printFile, '\n');

    save(opt.savefile,'feat');

otherwise
   error(sprintf('Unknown method to make feature vectors: opt.method = %s',...
   opt.method));
end


% if not stdout or stderr, close the file
if printFile > 2
   fclose(printFile);
end

end % featVecs
