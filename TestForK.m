correctClassRateMAT = []; 
probCorrectMAT = [];  
probMat = []; 

correctClassRateMAT2 = []; 
probCorrectMAT2 = [];  
probMat2 = []; 


correctClassRateMAT3 = []; 
probCorrectMAT3 = []; 
probMat3 = []; 

[G1,G2,G3]  = createGraph(.15);

for  h= 1:5:100
    h   
    values = []; 
    values1 = []; 
    values2 = []; 
    
 for j = 1:5
[correctClassRate, probCorrect,correctClassRate2, probCorrect2,correctClassRate3, probCorrect3] = crossValDistMatGraphs(h,G1);
values = [values, probCorrect];
values1 = [values1, probCorrect2];
values2 = [values2, probCorrect3];
 end
 
correctClassRateMAT = [correctClassRateMAT,correctClassRate]; 
probCorrectMAT = [probCorrectMAT; probCorrect];
probMat = [probMat;  mean(values) std(values)];

correctClassRateMAT2 = [correctClassRateMAT2,correctClassRate2]; 
probCorrectMAT2 = [probCorrectMAT2; probCorrect2];
probMat1 = [probMat1;  mean(values1) std(values1)];


correctClassRateMAT3 = [correctClassRateMAT3,correctClassRate3]; 
probCorrectMAT3 = [probCorrectMAT3; probCorrect3];
probMat2 = [probMat2;  mean(values2) std(values2)];

end


figure(1)
errorbar([1:10:110], probMat(:,1), probMat(:,2),'o')
      xlabel('Num of Eigenvectors'); ylabel('Probability Correct');
%       figure(2)
% errorbar([1:10:110], probCorrectMAT1(:,1), probCorrectMAT1(:,2),'o')
%       xlabel('Num of Eigenvectors'); ylabel('probCorrect');