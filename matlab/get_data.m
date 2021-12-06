clc;
close all;
clear;

frame_header = 24;
Disk = 'D:';
% Disk = '/mnt/DiskA-1.7T/ztz';
dataSource = [Disk, '/WIFI_Dataset/AugData', '/WiFi_Waveform'];
localList = {'P4'};

deviceCount = 11;
countPerDevice = 200;
for loop1 = 1:length(localList)
    FilePoint = char(localList(loop1));
    count = zeros(deviceCount, 1);
    index = 1;
    Temp_Store_Frame_Label = zeros(countPerDevice * deviceCount, 1);
    Temp_Store_Waveform = zeros(countPerDevice * deviceCount, 8020);
    namelist = dir([dataSource, '/', FilePoint, '/*']);
    fileNums = length(namelist);
    for i=1:fileNums
        file_name{i} = namelist(i).name;
    end
    for i = 3:fileNums
        DataRoot = [dataSource, '/', FilePoint, '/', file_name{i}];
        load(DataRoot);
        for frame = 1: length(Store_Frame_Label)
            frame_label = Store_Frame_Label(frame);
            if frame_label == 69
                frame_label = 11;
            end
            currentCount = count(frame_label);
            if currentCount >= 200
                continue;
            end
            Temp_Store_Waveform(index, :) = Store_Waveform(frame, :);
            Temp_Store_Frame_Label(index,:) = frame_label;
            count(frame_label) = count(frame_label) + 1;
            index = index + 1;
        end
    end
    Store_Waveform = Temp_Store_Waveform;
    Store_Frame_Label = Temp_Store_Frame_Label;
    save([FilePoint, '.mat'], 'Store_Waveform', 'Store_Frame_Label');
end
