clc;
close all;
clear;

global LONG_FREQUENCY_RESP;
global deviceIndex;
global N;
global Plot_Table_Size;
global Plot_Table_Max;
global Process_Data_Offset;
global B;
global dataTarget;

deviceIndex = zeros(11, 1);
LONG_FREQUENCY_RESP = transpose(STS_LTS_Generator('LONG-F'));
N = 90;
B = 320;
Plot_Table_Size = 31;  % DCTF Size
Plot_Table_Max = 1.4;  % DCTF Size
Process_Data_Offset = 0; % Delete some samples for generating DCTF

dataRoot = 'D:';
dataName = '20200520_WiFi_Data_Waveform';
dataSource = [dataRoot, '/WIFI_Dataset/', dataName];
dataTarget = [dataRoot, '/WIFI_Dataset/', dataName, '/DataAug'];
localList = {'P1', 'P2', 'P3', 'P4'};
snrList = {'no', '30'};
for local = 1:length(localList)
    FilePoint = char(localList(local));
    fileList = get_filename([dataSource, '/', FilePoint, '/*']);
    disp(['current point is ', FilePoint, ', file count is ', num2str(length(fileList))]);
    for channelSeed = 1: 3
        tgnChannel = wlanTGnChannel('RandomStream', 'mt19937ar with seed', ...
            'Seed', channelSeed, 'DelayProfile', 'Model-B');
        disp(['Seed: ', num2str(channelSeed)]);
        for s = 1:length(snrList)
            disp(['SNR: ', char(snrList(s))]);
            for i = 1: length(fileList)
                disp(['File: ', fileList(i)]);
                str = [dataSource, '/', FilePoint, '/', fileList(i)];
                load(cell2mat(strcat(str)));
                % Store_Frame_Label, Store_Waveform
                % 保证Store_Waveform每个列向量是一个信号
                Store_Waveform = transpose(Store_Waveform);
                Store_Frame_Label = transpose(Store_Frame_Label);
                for j = 1:length(Store_Frame_Label)
                    T = Store_Waveform(:, j);
                    T = addChannelToSignal(T, tgnChannel);
                    signalToImg(T, Store_Frame_Label(j), char(snrList(s)), ...
                        FilePoint, 1, num2str(channelSeed));
                end
            end
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
