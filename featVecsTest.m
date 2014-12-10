function [feat] = featVecsTest()
% compute the feature vectors for the test data set
% this is a simple driver for featVecs.m

[wavList, genreCode] = getTestData();

featVecOpt = struct('method','testWCH','savefile','featVecsTestWCH.mat');
featVecOpt = struct('method','testDale','savefile','featVecsTestDale.mat');
feat = featVecs(wavList, featVecOpt);

end
