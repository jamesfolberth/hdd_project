%savefile = 'featVecsWCH.mat';
saveFile = 'featVecsDale.mat';

load(saveFile);

if ~exist('feat')
   error('Feature matrix ''feat'' not found');
end

featFull = feat;

cvrep = 5;

% Reduce with genre specific rankings
nrows = [1 2 3 4 6 8 10 12 14 16 18 20 25 30 35 40 45 50 55 60 70 80 90 100 130 160 198];
%nrows = [1 2 4 6 8 10 12 14 20 25 30 35 40 45 50];

counts = zeros(length(nrows),11);
accm = zeros(length(nrows),11);
accs = zeros(length(nrows),11);

for(f = 0:0.1:1)
    f
    
    ranks = pageRankDimRed(featFull,struct('method','adjusted','factor',f));
    
    accg = zeros(length(nrows),cvrep);
    
    c = 0;
    for n = nrows
       c = c+1;
       fprintf(1,'\rrow = %d',n);
       ind = ranks(1:n,1:6);
       ind = unique(ind(:));
       feat = featFull(ind,:);

       save('featRedTemp.mat','feat');
       for k = 1:cvrep
            xvalOpt = struct('XValNum', 10, 'dimRed','none', 'kNNNum',5);
            [a,s] = crossValkNNFeatVec('featRedTemp.mat',xvalOpt);

            accg(c,k) = sum(diag(a)./reshape(sum(a,1), [6 1])*1/6);
        end
        counts(c,round(10*f+1)) = length(ind);
    end
    fprintf(1,'\n');

    accm(:,round(f*10+1)) = mean(accg,2);
    accs(:,round(f*10+1)) = std(accg,0,2);
end
    
figure;
errorbar(counts,accm,accg);
title('Genre Based PR')
xlabel('Dimension')
ylabel('Adj classification rate')
xlim([0 200])
ylim([.3 .7])
legend('0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0');
print('adjPR.pdf','-dpdf')

save('dimresults.mat','accm','accs','counts','mydims','nrows')
