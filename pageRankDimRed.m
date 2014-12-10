function ranks = pageRankDimRed(feat,opts)
% Takes a training feature matrix and returns a matrix of feature rankings
% Each genre is considered separately to attempt to grab the best features
% for that genre
% The entire collection is then considered to try to find the overall best
% features
% Squared spearman correlation is used for 


if nargin == 0
   %savefile = 'featVecsWCH.mat';
   saveFile = 'featVecsDale.mat';
   load(saveFile, '-mat');

   if ~exist('feat')
      error('Feature matrix ''feat'' not found in savefile: %s',saveFile);
   end
end

if nargin < 2
    %opts = struct('method','basic');
    opts = struct('method','adjusted','factor',0.3);
end

if strcmp(opts.method,'adjusted') && ~isfield(opts,'factor')
    opts.factor = 0.3;
end

% Indices for where the genre switches
% Note: Shouldn't be hardcoded
gInd = [0 320 434 460 505 607 729];

% Initialize the ranking matrix
% Columns correspond to genres
% Rows correspond to features
ranks = zeros(size(feat,1),7);

switch opts.method
    
    case 'basic'
        % For each genre, find the "best" features
        for n = 1:6
            rsq = corr(feat(:,gInd(n)+1:gInd(n+1))','type','Spearman').^2;
            rsq(isnan(rsq)) = 0;
            rsq = rsq - eye(size(feat,1));

            H = diag(1./sum(rsq,2))*rsq;

            [r,~] = eigs(H',1);
            r = -r;

            [~,I] = sort(r,'descend');
            ranks(:,n) = I;
        end

        % For the entire collection, find the "best" features
        rsq = corr(feat','type','Spearman').^2;
        rsq = rsq - eye(size(feat,1));

        H = diag(1./sum(rsq,2))*rsq;

        [r,~] = eigs(H',1);
        r = -r;

        [~,I] = sort(r,'descend');
        ranks(:,7) = I;

    case 'cross'
	W = cell(6,1);
	for n = 1:6
	    rsq = corr(feat(:,gInd(n)+1:gInd(n+1))','type','Spearman').^2;
	    rsq(isnan(rsq)) = 0;
	    rsq = rsq - eye(size(feat,1));
	    W{n} = rsq;
	end

	for n = 1:6
	    H = W{n} + opts.factor*(ones(size(W{n})) - (W{1}+W{2}+W{3}+W{4}+W{5}+W{6}-W{n}));
	    H = diag(1./sum(H,2))*H;

	    [r,~] = eigs(H',1);
	    r = -r;

	    [~,I] = sort(r,'descend');
	    ranks(:,n) = I;
	end

	% For the entire collection, find the "best" features
        rsq = corr(feat','type','Spearman').^2;
        rsq = rsq - eye(size(feat,1));

        H = diag(1./sum(rsq,2))*rsq;

        [r,~] = eigs(H',1);
        r = -r;

        [~,I] = sort(r,'descend');
        ranks(:,7) = I;

	    
    case 'adjusted'
        ranks = zeros(size(feat,1),7);
        
        for n = 1:6
            rsq = corr(feat(:,gInd(n)+1:gInd(n+1))','type','Spearman').^2;
            rsq(isnan(rsq)) = 0;
            rsq = rsq - eye(size(feat,1));

%             vInd = 0*ranks(:,1);
            rInd = 1:size(ranks,1);
            
            for(k = 1:(size(ranks,1)-1))
                H = diag(1./sum(rsq,2))*rsq;

                [r,~] = eigs(H',1);
                if(r(1) < 0)
                    r = -r;
                end

                [~,mI] = max(r);
%                 vInd(k) = mI;
%                 if(k > 1)
%                     ranks(k,n) = mI + sum(vInd(1:(k-1)) <= mI);
%                 else
%                     ranks(k,n) = mI;
%                 end
                
                ranks(k,n) = rInd(mI);
                rInd(mI) = [];
%                 vInd(k) = mI;
                
                %rsq = bsxfun(@times, rsq, (1-opts.factor*rsq(:,mI)));
                rsq = rsq*diag(1-opts.factor*rsq(:,mI));
                rsq(mI,:) = [];
                rsq(:,mI) = [];
            end
%             ranks(k+1,n) = 1 + sum(vInd(1:k) == 1);
            ranks(k+1,n) = rInd(1);
            
        end

        rsq = corr(feat','type','Spearman').^2;
        rsq = rsq - eye(size(feat,1));

%         vInd = 0*ranks(:,1);
        rInd = 1:size(ranks,1);
        
        for(k = 1:size(ranks,1)-1)
            H = diag(1./sum(rsq,2))*rsq;

            [r,~] = eigs(H',1);
            if(r(1) < 0)
                r = -r;
            end

            [~,mI] = max(r);
%             vInd(k) = mI;
%             if(k > 1)
%                 ranks(k,7) = mI + sum(vInd(1:(k-1)) <= mI);
%             else
%                 ranks(k,7) = mI;
%             end

            ranks(k,n) = rInd(mI);
            rInd(mI) = [];

            %rsq = bsxfun(@times, rsq, (1-opts.factor*rsq(:,mI)));
            rsq = rsq*diag(1-opts.factor*rsq(:,mI));
            rsq(mI,:) = [];
            rsq(:,mI) = [];
        end
%         ranks(k+1,7) = 1 + sum(vInd(1:k) == 1);
        ranks(k+1,7) = rInd(1);
        
    otherwise
        error('Unknown PageRank method: %s', opt.method);
end

end
