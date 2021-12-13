clc
close all;
clear;

%% 将不同的设备划分到同一个文件中
% SP = 'P1';
% for iDevice = 1:10
%     load(['D:\WIFI_Dataset\AugData\ClearedDataset-1\', SP, '\', SP, '.mat']);
%     [m, n] = size(Store_Waveform);
%     waveform = zeros(200, n);
%     label = zeros(200, 1);
%     index = 1;
%     for iLabel = 1:m
%         if Store_Frame_Label(iLabel) == iDevice
%             waveform(index, :) = Store_Waveform(iLabel, :);
%             label(index, :) = iDevice;
%             index = index + 1;
%         end
%     end
%     Store_Waveform = waveform;
%     Store_Frame_Label = label;
%     saveDir = 'D:\WIFI_Dataset\AugData\ClearedDataset-1-RawSlice';
%     saveFile = [SP, '_D', num2str(iDevice), '.mat'];
%     save([saveDir, '\', saveFile], 'Store_Waveform', 'Store_Frame_Label');
%     disp([saveDir, '\', saveFile, '  has been stored at disk']);
% end

%% 从指定的数据中slice_sampling.m
dataSource = 'D:\WIFI_Dataset\AugData\ClearedDataset-1-RawSlice';
devices = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10"];
points = ["P2"];
for p = 1:length(points)
    for d = 1:length(devices)
        file_name = [char(points(p)), '_', char(devices(d)), '.mat'];
        sourceFile = [dataSource, '\', 'PointAndDevice', '\', file_name];
        targetFile = [dataSource, '\', 'Slice', '\', 'Slice10_', file_name];
        slice_sampling(sourceFile, targetFile, 128, 10);
    end
end
