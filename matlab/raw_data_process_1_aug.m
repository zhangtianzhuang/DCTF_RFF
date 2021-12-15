clc;
clear;
close all;

dataSource = 'D:\WIFI_Dataset\AugData\ClearedDataset-1-RawSlice';
devices = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10"];
points = ["P1", "P2"];
% 如果channelSeed = 0 表示信号不通过仿真信道
channelSeedFrom = 1;
channelSeedTo = 1;
for channelSeed = channelSeedFrom:channelSeedTo
    tgnChannel = wlanTGnChannel('RandomStream', 'mt19937ar with seed', ...
                'Seed', channelSeed, 'DelayProfile', 'Model-B');
    for p = 1:length(points)
        for d = 1:length(devices)
            file_name = [char(points(p)), '_', char(devices(d)), '.mat'];
            sourceFile = [dataSource, '\', 'PointAndDevice', '\', file_name];
            targetDir = [dataSource, '\', 'Augmentation'];
            mkdirIfNotExist(targetDir);
            
            load(sourceFile, 'Store_Waveform', 'Store_Frame_Label');
            if channelSeed > 0
                % 如果维度不满足，进行修改
                if iscolumn(Store_Frame_Label)
                    Store_Waveform = transpose(Store_Waveform);
                    Store_Frame_Label = transpose(Store_Frame_Label);
                end
                % 原始信号，通过仿真信道
                Store_Waveform = addChannelToSignal(Store_Waveform, tgnChannel);
                % 修改信号格式，每一行为一个信号
                Store_Waveform = transpose(Store_Waveform);
                Store_Frame_Label = transpose(Store_Frame_Label);
                % remove the signal delay
                Store_Waveform(:,1:end-6) = Store_Waveform(:,7:end);
            end
            channelSeedStr = num2str(channelSeed);
            if channelSeed == 0
                channelSeedStr = 'no';
            end
            % 存储文件
            targetFile = [file_name(1:end-4), '_A', channelSeedStr, '.mat'];
            save([targetDir, '\', targetFile], 'Store_Waveform', 'Store_Frame_Label');
            disp([targetFile, ' has been finished!']);
        end
    end
end