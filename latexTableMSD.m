function latexTable(mu,sd, filename, fmt1, fmt2, names)
% print mean and standard deviation like mu \pm sd

if nargin < 3
   filename = 'table.tex';
end

if nargin < 4
   fmt1 = '%3.2f';
   fmt2 = '%3.2f';
end

if nargin < 6
   names = {};
end

fid = fopen(filename,'w');

if fid == -1
   warning('fopen could not open file: %s',filename);
end

assert(size(mu,1) == size(sd,1) && size(mu,2) == size(sd,2));

%fprintf(fid,'\\documentclass{article}\n');
%fprintf(fid,'\\begin{document}\n');
fprintf(fid,'\\begin{tabular}{ ');

fprintf(fid, 'l||');
for col=2:size(mu,2)
   fprintf(fid,'l | ');
end
if ~isempty(names)
   fprintf(fid, 'l | ');
end
fprintf(fid,'}\n');

if ~isempty(names)
   for c = 1:size(mu,2)
      fprintf(fid, '& %s', strrep(names{c},'_','\_'));
   end
   fprintf(fid, '\\\\\\hline\n');
end

% now write the elements of the matrix
for r=1:size(mu,1)
   if ~isempty(names)
      fprintf(fid,'%s & ', strrep(names{r},'_','\_'));
   end
   for c=1:size(mu,2)
        if c==size(mu,2)
            fprintf(fid,strcat('$',fmt1,' \\pm ',fmt2,'$'),mu(r,c),sd(r,c));
        else
            fprintf(fid,strcat('$',fmt1,' \\pm ',fmt2,'$ & '),mu(r,c),sd(r,c));
        end
    end            

    fprintf(fid,' \\\\ \\hline \n');
end

%fprintf(fid,'\\hline\n');
fprintf(fid,'\\end{tabular}\n');
%fprintf(fid,'\\end{document}\n');
fclose(fid);

end 
