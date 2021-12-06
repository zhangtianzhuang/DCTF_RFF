function mkdirIfNotExist(path)
    if exist(path, 'dir') == 0
        mkdir(path);
    end
end

