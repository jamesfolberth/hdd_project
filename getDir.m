function dataDir = getDir()

if isunix
    [~, userName] = system('whoami');
    userName = strtrim(userName);
elseif ispc
    [~, userName] = system('echo %USERDOMAIN%\%USERNAME%');
end

%disp(userName)

if ( strcmp(userName,'dalekj') )
  dataDir = '/media/removable/SDcard/cd_data/';
elseif ( strcmp(userName,'james') )
  dataDir = './cd_data/';
elseif ( strcmp(userName,'aly') ) %TODO needs correct username here
  dataDir = 'unknown';
else
   error('Unknown user name.')
end

end
