function dataDir = getDir()

if isunix
    [~, userName] = system('whoami');
    userName = strtrim(userName);
elseif ispc
    [~, userName] = system('echo %USERDOMAIN%\%USERNAME%');
end

disp(userName)

if(userName == 'dalekj')
  dataDir = '/media/removable/SDcard/cd_data/';
elseif(userName == 'james') % needs correct username here
  dataDir = './cd_data/';
elseif(userName == 'aly') % needs correct username here
  dataDir = 'unknown';
end

end