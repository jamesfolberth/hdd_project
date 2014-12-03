opt1 = load('probCorrectopt2k_2.mat');
opt1 = opt1.probCorrectMAT; 
opt2 = load('probCorrectopt1k_5.mat');
opt2 = opt2.probCorrectMAT; 

opt3 = load('probCorrectopt3n2.mat');
opt3 = opt3.probCorrectMAT1; 

figure(1)

errorbar([1:10:110],opt3(:,1), opt3(:,2),'o')
      xlabel('Num of Eigenvectors'); ylabel('probCorrect');