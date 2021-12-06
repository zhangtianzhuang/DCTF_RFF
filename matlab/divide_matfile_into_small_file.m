clc;
close all;
clear;

Disk = 'D:/WIFI_Dataset/AugData/';
% Disk = '/mnt/DiskA-1.7T/ztz/WIFI_Dataset/';
dataSource = [Disk, 'ClearedDataset-1'];
% dataSource = ['D:\WIFI_Dataset\20200520_WiFi_Data_Waveform\'];
localList = {'P4'};
countPerFile = 20;
sampleLength = 8020;
fileCountPerPoint = 2000 / countPerFile;
for loop1 = 1:length(localList)
   FilePoint = char(localList(loop1));
   load([dataSource, '/', FilePoint, '/', FilePoint, '.mat']);
   Temp_Store_Frame_Label = Store_Frame_Label;
   Temp_Store_Waveform = Store_Waveform;
   fileArrayWave = cell(fileCountPerPoint, 1);
   fileArrayLabel = cell(fileCountPerPoint, 1);
   fileIndex = zeros(fileCountPerPoint, 1);
   deviceIndex = zeros(10, 1);
   for i = 1:length(fileArrayWave)
       fileArrayWave{i} = zeros(countPerFile, sampleLength);
       fileArrayLabel{i} = zeros(countPerFile, 1);
   end
   for i = 1:length(Temp_Store_Frame_Label)
       label = Temp_Store_Frame_Label(i);
       if label > 10
           continue;
       end
       deviceIndex(label) = deviceIndex(label) + 1;
       whichFile = floor((deviceIndex(label) + 1)/2);
       fileIndex(whichFile) = fileIndex(whichFile) + 1;
       fileArrayWave{whichFile}(fileIndex(whichFile),:) = Temp_Store_Waveform(i, :);
       fileArrayLabel{whichFile}(fileIndex(whichFile)) = label;
   end
   for i = 1:length(fileArrayWave)
       Store_Waveform = fileArrayWave{i};
       Store_Frame_Label = fileArrayLabel{i};
       targetDir = [Disk, 'ClearedDataset-', num2str(countPerFile), '/', FilePoint];
       mkdirIfNotExist(targetDir);
       targetPath = [targetDir, '/', FilePoint, '-', num2str(i), '.mat'];
       save(targetPath, 'Store_Waveform', 'Store_Frame_Label');
       disp(['write file:', targetPath]);
   end
end
