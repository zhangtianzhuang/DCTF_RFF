clc;
clear;
close all;

dataSource = 'D:\WIFI_Dataset\AugData\ClearedDataset-1-RawSlice';
devices = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10"];
points = ["P1", "P2"];
seeds = ["A1"];
snrs = [5, 10, 15, 20, 25, 30 100];  % 100表示不添加噪声 %
slice_size = 128;
slice_number = 10;
for p = 1:length(points)
    for d = 1:length(devices)
        for c = 1:length(seeds)
            for s = 1:length(snrs)
                snr = num2str(snrs(s));
                if snrs(s) == 100
                    snr = 'no';
                end
                file_name = [char(points(p)), '_', char(devices(d)), '_', char(seeds(c)), '_S', snr, '.mat'];
                sourceFile = [dataSource, '\', 'SNR', '\', file_name];
                targetDir = [dataSource, '\Slice\', 'Slice', num2str(slice_number),'-', num2str(slice_size)];
                mkdirIfNotExist(targetDir);
                targetFile = [file_name(1:end-4), '_L', num2str(slice_number),'-', ...
                    num2str(128), '.mat'];
                slice_sampling(sourceFile, [targetDir, '\', targetFile], slice_number, slice_size);
                disp(targetFile);
            end
        end
    end
end