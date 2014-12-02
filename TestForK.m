correctClassRateMAT = []; 
probCorrectMAT = [];  
for k = 6:2:20
[correctClassRate, probCorrect] = crossValDistMatGraphs(k);
correctClassRateMAT = [correctClassRateMAT,correctClassRate]; 
probCorrectMAT = [probCorrectMAT; probCorrect];
end