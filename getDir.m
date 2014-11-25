function dataDir = getDir()

if isunix
    [~, userName] = system('whoami');
    userName = strtrim(userName);
elseif ispc
    [~, userName] = system('echo %USERNAME%');
end

%disp(userName)

if ( strcmp(userName,'dalekj') )
  dataDir = '/media/removable/SDcard/cd_data/';
elseif ( strcmp(userName,'james') )
  dataDir = './cd_data/';
elseif ( strcmp(userName,'alfox') ) 
  dataDir = '/Users/alfox/Documents/Data/';
elseif ( strcmp(deblank(userName),'Dale') )
    dataDir = '../cd_data/';
else
   error('Unknown user name.')
end

end
