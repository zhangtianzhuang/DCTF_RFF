clc;
clear;
close all;

dataSource = 'D:\WIFI_Dataset\AugData\ClearedDataset-1-RawSlice';
devices = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10"];
points = ["P1", "P2"];
seeds = ["A1"]; % , "A1", "A2"
snrs = [5, 10, 15, 20, 25, 30, 100];  % 100表示不添加噪声 % 25, 30, 100
for p = 1:length(points)
    for d = 1:length(devices)
        for c = 1:length(seeds)
            for s = 1:length(snrs)
                file_name = [char(points(p)), '_', char(devices(d)), '_', char(seeds(c)), '.mat'];
                sourceFile = [dataSource, '\', 'Augmentation', '\', file_name];
                targetDir = [dataSource, '\', 'SNR'];
                mkdirIfNotExist(targetDir);
                load(sourceFile, 'Store_Waveform', 'Store_Frame_Label');
                if snrs(s) < 100
                    [m, n] = size(Store_Waveform);
                    for k = 1:m
                        Store_Waveform(k, :) = awgn(Store_Waveform(k, :), snrs(s), 'measured');
                    end
                end
                snr = num2str(snrs(s));
                if snrs(s) == 100
                    snr = 'no';
                end
                % 存储文件
                targetFile = [file_name(1:end-4), '_S', snr, '.mat'];
                save([targetDir, '\', targetFile], 'Store_Waveform', 'Store_Frame_Label');
                disp(targetFile);
            end
        end
    end
end