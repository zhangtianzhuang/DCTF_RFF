clear;
close all;
clc;

%SystemType = 'windows';
%SystemType = 'linux';
% isLocal = 0;
isLocal = 1;

% windows config
% if strcmp(SystemType, 'windows')
%     dataRoot = 'D:';
if isLocal == 1
    dataRoot = 'D:';
else
% linux config
    dataRoot = '/mnt/DiskA-1.7T/ztz';
end

% General Config
dataSource = [dataRoot, '/WIFI_Dataset/AugData/', 'ClearedDataset-1'];
dataTarget = [dataRoot, '/WIFI_Dataset/AugData/', 'ClearedDataset-1_ModelB_S6-10'];
localList = {'P1', 'P2'}; % 'P1', 'P2', 'P3', 'P4'
modelType = 'Model-B';

for local = 1:length(localList)
    FilePoint = char(localList(local));
    pointName = [dataSource, '/', FilePoint, '/*'];
    fileList = get_filename(pointName);
    disp(['current point is ', pointName, ', the number of file is ', num2str(length(fileList))]);
    for channelSeed = 6:10
        tgnChannel = wlanTGnChannel('RandomStream', 'mt19937ar with seed', ...
            'Seed', channelSeed, 'DelayProfile', modelType);
        for i = 1: length(fileList)
            curFileName = char(fileList(i));
            disp(['File: ', curFileName]);
            str = [dataSource, '/', FilePoint, '/', curFileName];
            load(str);
            % Store_Frame_Label, Store_Waveform
            if iscolumn(Store_Frame_Label)
                Store_Waveform = transpose(Store_Waveform);
                Store_Frame_Label = transpose(Store_Frame_Label);
            end
            % 
            Store_Waveform = addChannelToSignal(Store_Waveform, tgnChannel);
            
            Store_Waveform = transpose(Store_Waveform);
            Store_Frame_Label = transpose(Store_Frame_Label);
            % remove the signal delay
            Store_Waveform(:,1:end-6) = Store_Waveform(:,7:end);
            % 
            targetDirName = [dataTarget, '/', FilePoint];
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


function signal = addChannelToSignal(signal, channel)
    [~, n] = size(signal);
    for i = 1:n
        x = signal(:, i);
        y = channel(x);
        signal(:, i) = y;
    end
end