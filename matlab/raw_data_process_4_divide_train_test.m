close all;
clc;
clear;

dataSource = 'D:/WIFI_Dataset/AugData/ClearedDataset-1-RawSlice';
devices = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10"];
points = ["P1", "P2"];
seeds = ["A1"];
snrs = [5, 10, 15, 20, 25, 30, 100];  % 100表示不添加噪声 %
slice_size = 128;
slice_number = 10;
ratio = 4; % 训练集样本数与测试集样本数的比例

for p = 1:length(points)
    for d = 1:length(devices)
        for c = 1:length(seeds)
            for s = 1:length(snrs)
                snr = num2str(snrs(s));
                if snrs(s) == 100
                    snr = 'no';
                end
                file_name = [char(points(p)), '_', char(devices(d)), '_', char(seeds(c)), '_S', snr, ...
                    '_', 'L', num2str(slice_number),'-', num2str(128), '.mat'];
                sourceFile = [dataSource, '/Slice/', 'Slice', num2str(slice_number),'-', ...
                    num2str(slice_size), '/', file_name];
                trainTargetDir = [dataSource, '/Train/', 'Slice', num2str(slice_number),'-', ...
                    num2str(slice_size)];
                testTargetDir = [dataSource, '/Test/', 'Slice', num2str(slice_number),'-', ...
                    num2str(slice_size)];
                mkdirIfNotExist(trainTargetDir);
                mkdirIfNotExist(testTargetDir);
                % 加载源数据
                load(sourceFile, 'Store_Waveform', 'Store_Frame_Label');
                [m, n] = size(Store_Waveform);
                ran = randperm(m);
                boundary = m * ratio / (ratio+1);
                train_size = boundary;
                test_size = m - boundary;

                waveform = Store_Waveform;
                label = Store_Frame_Label;

                train_waveform = zeros(train_size, n);
                train_label = zeros(train_size, n);
                train_index = 1;
                for k = 1:boundary
                    idx = ran(k);
                    train_waveform(train_index, :) = waveform(idx, :);
                    train_label(train_index, :) = label(idx, :);
                    train_index = train_index+1;
                end
                Store_Waveform = train_waveform;
                Store_Frame_Label = train_label;
                trainTargetFile = [file_name(1:end-4), '_Train.mat'];
                save([trainTargetDir, '/', trainTargetFile], 'Store_Waveform', 'Store_Frame_Label');
                disp(trainTargetFile);

                test_waveform = zeros(test_size, n);
                test_label = zeros(test_size, n);
                test_index = 1;
                for k = boundary+1:m
                    idx = ran(k);
                    test_waveform(test_index, :) = waveform(idx, :);
                    test_label(test_index, :) = label(idx, :);
                    test_index = test_index + 1;
                end
                Store_Waveform = test_waveform;
                Store_Frame_Label = test_label;
                testTargetFile = [file_name(1:end-4), '_Test.mat'];
                save([testTargetDir, '/', testTargetFile], 'Store_Waveform', 'Store_Frame_Label');
                disp(testTargetFile);
            end
        end
    end
end
