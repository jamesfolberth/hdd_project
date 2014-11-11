function [fpMed, fpAll] = flucPat(melPowDB,opt)
% Compute various fluctuation patterns
% See Section 2.2.4 of Pampalk 2006 for details

persistent t flux filt1 filt2

% If t isn't defined yet, define it and the other constants
% t     - used to split the 36 Mel bands in to 12 or so frequency bands
% flux  - fluctuation strength weights
% filt1 - filter to smooth over modulation
% filt2 - filter to smooth over frequencies
if ( isempty(t) )
   t = zeros([1 36]);

   % The choice of t is more or less arbitrary
   % this is the t given in Pampalk 2006
   t(1) = 1; t(2) = 2; t(3:4) = 3; t(5:6) = 4;
   t(7:8) = 5; t(9:10) = 6; t(11:12) = 7; t(13:14) = 8;
   t(15:18) = 9; t(19:23) = 10; t(24:29) = 11; t(30:36) = 12;

   f = linspace(0, 11025/512/2, 64+1);
   flux = repmat(1./(f(2:32)/4 + 4./f(2:32)), max(t), 1);
   w = [0.05 0.1 0.25 0.5 1 0.5 0.25 0.1 0.05];
   filt1 = filter2(w, eye(max(t)));
   filt1 = filt1./repmat(sum(filt1,2), 1, max(t));
   filt2 = filter2(w, eye(30)); % don't know where this 30 comes from
   filt2 = filt2./repmat(sum(filt2,2), 1, 30);
end

% Reduce melPowDB to the frequency bins defined by t
mel2 = zeros([max(t) size(melPowDB,2)]);
for i=1:max(t)
   mel2(i,:) = sum(melPowDB(t==i,:),1);
end

% We shouldn't be dealing with extremely short tracks
% pad with zeros if necessary
if size(mel2,2) < 128
   mel2 = [zeros(max(t), ceil(128-size(mel2,2))) mel2]; 
end

% Compute and smooth frequency and amplitude modulation
numSegments = floor( (size(mel2,2) - 128)/64 + 1);
fpAll = zeros([numSegments max(t)*30]);
for i=1:numSegments
   X = fft(mel2(:,(1:128)+64*(i-1)), 128, 2); % 128-point FFT
   X2 = abs(X(:,2:32)).*flux; % amplitude spectrum
   X2 = filt1*abs(diff(X2,1,2))*filt2;
   fpAll(i,:) = transpose(X2(:));
end

% Summarize all fluctuation patterns 
fpMed = median(fpAll,1);

% Note that we can compare two songs based on their fluctuation patterns
% by computing the 2-norm of the difference
%    d = norm(fp1 - fp2, 2);

