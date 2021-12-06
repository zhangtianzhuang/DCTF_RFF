clear;
close all;
clc;

%SystemType = 'windows';
   SystemType = 'linux';

% windows config
if strcmp(SystemType, 'windows')
    dataRoot = 'D:';
else
% linux config
    dataRoot = '/home/bjtu/ztz';
end

% General Config
%dataName = 'WiFi_Data_Waveform_Filtered';
dataSource = [dataRoot, '/WIFI_Dataset/', '20200520_WiFi_Data_Waveform'];
dataTarget = [dataRoot, '/WIFI_Dataset/', 'WiFi_Data_Waveform_Augmentation/SimulationPoint'];
localList = {'P1'}; % 'P1', 'P2', 'P3', 'P4'
modelType = 'Model-D';

for local = 1:length(localList)
    FilePoint = char(localList(local));
    pointName = [dataSource, '/', FilePoint, '/*'];
    fileList = get_filename(pointName);
    disp(['current point is ', pointName, ', file count is ', num2str(length(fileList))]);
    for channelSeed = 1:20
        tgnChannel = wlanTGnChannel('RandomStream', 'mt19937ar with seed', ...
            'Seed', channelSeed, 'DelayProfile', modelType);
        for i = 1: length(fileList)
            curFileName = char(fileList(i));
            disp(['File: ', curFileName]);
            str = [dataSource, '/', FilePoint, '/', curFileName];
            load(str);
            % Store_Frame_Label, Store_Waveform
            Store_Waveform = transpose(Store_Waveform);
            Store_Frame_Label = transpose(Store_Frame_Label);
            % 将Store_Waveform的每一列，经过一个信道
            Store_Waveform = addChannelToSignal(Store_Waveform, tgnChannel);
            targetDirName = [dataTarget, '/', 'P4'];
            if exist(targetDirName, 'dir') == 0
                mkdir(targetDirName);
            end
            targetFileName = [ '/', modelType, '_C-', ...
                num2str(channelSeed), '_', curFileName(1:end-4), '.mat'];
            targetName = [targetDirName, targetFileName];
            save(targetName, 'Store_Waveform', 'Store_Frame_Label');
            disp(['done: ', targetName]);
        end
    end
end

function res = addChannelToSignal(signal, channel)
    [~, n] = size(signal);
    for i = 1:n
        x = signal(:, i);
        y = channel(x);
        signal(:, i) = y;
    end
    res = signal;
end