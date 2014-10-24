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
   if ( isOctave() )
      [wav,fs] = wavread(wavFile);
   else
      [wav,fs] = audioread(wavFile,'double');
   end
   
   wav = wav*10^(96/20);

   computePowerOpt = struct('segLength',512,'shiftLength',512*0.5);
   [pow, segLength, shiftLength] = computePower(wav,computePowerOpt);

   mfccOpt = struct('method','dct','numCoeffs',20);
   [melPowCoeffs, melPowDB] = mfcc(wav, fs, mfccOpt.method, mfccOpt.numCoeffs);


   % pad with zeros to recover Mel power spectrum
   recovMelPowDB = [melPowCoeffs; ...
                    zeros([36-mfccOpt.numCoeffs size(melPowCoeffs,2)])];
   recovMelPowDB = idct(recovMelPowDB);

   plotSpectra(pow, melPowDB, recovMelPowDB);

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

   subplot(3,1,2)
   imagesc(melPowDB)
   colorbar()
   set(gca,'YDir','normal')
   set(gca,'xtick',[])

   subplot(3,1,3)
   imagesc(recovMelPowDB)
   colorbar()
   set(gca,'YDir','normal')
   set(gca,'xtick',[])
end
