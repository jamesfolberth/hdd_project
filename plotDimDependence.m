%savefile = 'featVecsWCH.mat';
saveFile = 'featVecsDale.mat';

load(saveFile);

if ~exist('feat')
   error('Feature matrix ''feat'' not found');
end

featFull = feat;

cvrep = 5;

%ranks = pageRankDimRed(featFull,struct('method','basic'));
ranks = pageRankDimRed(featFull,struct('method','adjusted','factor',0.3));

% % Reduce with the overall rankings
% %mydims = [1 2 4 6 8 10 15 20 25 30 40 50 60 70 80 100 120 140 160 180 198];
% mydims = [1 2 4 6 8 10 15 20 25 30 40];
% acc = zeros(length(mydims),cvrep);
% c = 0;
% for n = mydims
%     c = c+1;
%     fprintf(1,'\rdim = %d',n);
%     ind = ranks(1:n,7);
%     ind = unique(ind(:));
%     feat = featFull(ind,:);
% 
%     save('featRedTemp.mat','feat');
%     for k = 1:cvrep
%         xvalOpt = struct('XValNum', 10, 'dimRed','none', 'kNNNum',5);
%         [a,s] = crossValkNNFeatVec('featRedTemp.mat',xvalOpt);
% 
%         acc(c,k) = sum(diag(a)./reshape(sum(a,1), [6 1])*1/6);
%     end
% end
% fprintf(1,'\n');
% 
% figure;
% %plot(mydims,mean(acc,2),'bo-')
% errorbar(mydims,mean(acc,2),std(acc,0,2));
% title('Full Dataset Based PR')
% xlabel('Dimension')
% ylabel('Adj classification rate')
% xlim([0 200])
% ylim([.3 .7])
% print('fullPR.pdf','-dpdf')


% Reduce with genre specific rankings
nrows = [1 2 3 4 6 8 10 12 14 16 18 20 25 30 35 40 45 50 55 60 70 80 90 100 130 160 198];
%nrows = [1 2 4 6 8 10 12 14 20 25 30 35 40 45 50];
accg = zeros(length(nrows),cvrep);
counts = zeros(length(nrows),1);
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
    counts(c) = length(ind);
end
fprintf(1,'\n');

figure;
%plot(counts,mean(accg,2),'bo-')
errorbar(counts,mean(accg,2),std(accg,0,2));
title('Genre Based PR')
xlabel('Dimension')
ylabel('Adj classification rate')
xlim([0 200])
ylim([.3 .7])
print('genrePR.pdf','-dpdf')

save('dimresults.mat','acc','accg','counts','mydims','nrows')
