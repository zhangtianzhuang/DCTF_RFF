clear;
close all
clc;
% 针对每一个文件，随机地分成两份，一份为训练集，一份为测试，比例
dataSource = 'D:\WIFI_Dataset\AugData\ClearedDataset-1-RawSlice\Slice';
trainDataTarget = 'D:\WIFI_Dataset\AugData\ClearedDataset-1-RawSlice\Train';
testDataTarget = 'D:\WIFI_Dataset\AugData\ClearedDataset-1-RawSlice\Test';
ratio = 4;  % 训练集样本数与测试集样本数的比例
files = get_filename(dataSource);
devices = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10"];
points = ["P2"];
for p = 1:length(points)
    for d = 1:length(devices)
        file_name = ['Slice10_', char(points(p)), '_', char(devices(d)), '.mat'];
        load([dataSource, '\', file_name], 'Store_Waveform', 'Store_Frame_Label');
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
        save([trainDataTarget, '\', 'Train_', file_name], 'Store_Waveform', 'Store_Frame_Label');

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
        save([testDataTarget, '\', 'Test_', file_name], 'Store_Waveform', 'Store_Frame_Label');
    end
end
