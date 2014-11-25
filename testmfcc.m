function [] = testmfcc()
% Function used to draw a song and run various tests with mfcc.m

   % pick a random song
   dataDir = getDir();
   [wavList,genre] = textread([dataDir,'ground_truth.csv'],'%s %s','delimiter',',');
   nSongs = length(wavList);
   % Fix the names
   wavList = strrep( wavList, '"', '');
   wavList = strrep( wavList, 'mp3','wav');
   wavFile = strcat(dataDir, wavList{randi(nSongs)});

   % read in the song
   if ( ~exist('audioread') )
      [wav,fs] = wavread(wavFile);
   else
      [wav,fs] = audioread(wavFile,'double');
   end
   
   wav = wav*10^(96/20);

   %computePowerOpt = struct('segLength',512,'shiftLength',512*0.5);
   %[pow, segLength, shiftLength] = computePower(wav,computePowerOpt);

   %mfccOpt = struct('method','dct','numTerms',20);
   %%mfccOpt = struct('method','wav','wName','sym2','wLevel',5,'numTerms',20);
   %[compMelPow, melPowDB] = mfcc(wav, fs, mfccOpt);

   %%mfccOpt = struct('method','dct','numTerms',20);
   %mfccOpt = struct('method','wav','wName','db3','wLevel',5,'numTerms',5);
   %[compMelPow, melPowDB] = mfcc(wav, fs, mfccOpt);

   %recovMelPowDB = recoverMelPower(compMelPow, mfccOpt);

   %plotSpectra(pow, melPowDB, recovMelPowDB);

   mfccOpt1 = struct('method','dct','numTerms',20);
   mfccOpt2 = struct('method','wav','wName','sym5','wLevel',5,'numTerms',10);
   plotSpectraCompare(wav, fs, mfccOpt1, mfccOpt2);

end

function [] = plotSpectra(pow,melPowDB,recovMelPowDB)
% plot the DFT power spectrum (in dB), Mel power spectrum (in dB),
% and recovered Mel power spectrum (in dB)

   subplot(3,1,1)
   pow(pow < 1) = 1;
   imagesc(10*log10(pow)) % convert power to dB
   colorbar()
   set(gca,'YDir','normal')
   set(gca,'xtick',[])
   title('Power Spectrum (dB)')

   subplot(3,1,2)
   imagesc(melPowDB)
   colorbar()
   set(gca,'YDir','normal')
   set(gca,'xtick',[])
   title('Mel Power Spectrum (dB)')

   subplot(3,1,3)
   imagesc(recovMelPowDB)
   colorbar()
   set(gca,'YDir','normal')
   set(gca,'xtick',[])
   title('Reconstructed Mel Power Spectrum (dB)');
end

function [] = plotSpectraCompare(wav,fs,mfccOpt1,mfccOpt2)
% plot the Mel power spectrum (in dB) and the recovered Mel power spectrum 
% using two methods 

   %computePowerOpt = struct('segLength',512,'shiftLength',512*0.5);
   %[pow, segLength, shiftLength] = computePower(wav,computePowerOpt);

   [compMelPow1, melPowDB] = mfcc(wav, fs, mfccOpt1);
   [compMelPow2] = mfcc(wav, fs, mfccOpt2);


   subplot(3,1,1)
   imagesc(melPowDB)
   colorbar()
   set(gca,'YDir','normal')
   set(gca,'xtick',[])
   title('Mel Power Spectrum (dB)')

   recovMelPowDB1 = recoverMelPower(compMelPow1, mfccOpt1);
   subplot(3,1,2)
   imagesc(recovMelPowDB1);
   colorbar()
   set(gca,'YDir','normal')
   set(gca,'xtick',[])
   title(sprintf('Reconstructed Mel Power Spectrum (dB) - %s',...
      mfccOpt1.method));

   recovMelPowDB2 = recoverMelPower(compMelPow2, mfccOpt2);
   subplot(3,1,3)
   imagesc(recovMelPowDB2);
   colorbar()
   set(gca,'YDir','normal')
   set(gca,'xtick',[])
   title(sprintf('Reconstructed Mel Power Spectrum (dB) - %s',...
      mfccOpt2.method));

end
