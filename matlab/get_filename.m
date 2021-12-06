function fileList = get_filename(dirName)
    matFileList = dir(dirName);
    matFileListSize = length(matFileList);
    matFileName = repmat("", matFileListSize, 1);
    for i = 1:matFileListSize
        matFileName{i} = matFileList(i).name;
    end
    fileList = matFileName(3:end);
end