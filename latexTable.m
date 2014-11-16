function latexTable(matrix, filename, fmt, names)

if nargin < 2
   filename = 'table.tex';
end

if nargin < 3
   fmt = '%3.2f';
end

if nargin < 4
   names = {};
end

fid = fopen(filename,'w');

if fid == -1
   warning('fopen could not open file: %s',filename);
end

%fprintf(fid,'\\documentclass{article}\n');
%fprintf(fid,'\\begin{document}\n');
fprintf(fid,' \\begin{tabular}{ ');

fprintf(fid, 'l||');
for col=2:size(matrix,2)
   fprintf(fid,'l | ');
end
if ~isempty(names)
   fprintf(fid, 'l | ');
end
fprintf(fid,'}\n');

if ~isempty(names)
   for c = 1:size(matrix,2)
      fprintf(fid, ' & %s', strrep(names{c},'_','\_'));
   end
   fprintf(fid, '\\\\\\hline\n');
end

% now write the elements of the matrix
for r=1:size(matrix,1)
   if ~isempty(names)
      fprintf(fid,'%s & ', strrep(names{r},'_','\_'));
   end
   for c=1:size(matrix,2)
        if c==size(matrix,2)
            fprintf(fid,fmt,matrix(r,c));
        else
            fprintf(fid,strcat(fmt,' & '),matrix(r,c));
        end
    end            

    fprintf(fid,' \\\\ \\hline \n');
end

%fprintf(fid,'\\hline\n');
fprintf(fid,'\\end{tabular}\n');
%fprintf(fid,'\\end{document}\n');
fclose(fid);

end 
