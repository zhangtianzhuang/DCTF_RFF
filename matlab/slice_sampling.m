%% 从给定的数据集中抽取slice作为训练样本
% 输入参数：
%       原始数据以及设备标签 --> Store_Waveform, Store_Frame_Label
%       每个样本抽取样本个数 --> number_sample
%       每个样本的提取的起始位置 --> index
%       slice的长度，也可以通过index计算，但是为了方便，提供该参数 --> slice_length
% 输出参数：
%       提取的slice数据
function slice_sampling(source, target, slice_number, slice_size)
%     load('D:\WIFI_Dataset\AugData\ClearedDataset-1\P2\P2.mat');
    load(source, 'Store_Waveform', 'Store_Frame_Label');
    offset = 24;
%     slice_length = 128;
%     number_sample = 10;
    index = zeros(slice_number, 2);
    index(:, 1) = offset : slice_size : offset + slice_size * slice_number - 1;
    index(:, 2) = index(:, 1) + slice_size - 1;
    % 输出的数据
    [m, ~] = size(Store_Waveform); % m表示有多少个样本
    data_waveform = zeros(m * slice_number, slice_size);
    data_label = zeros(m * slice_number, 1);
    for iSample = 1: m  % 遍历所有的样本
        curSample = Store_Waveform(iSample, :);
        slices = get_slice_from_one_sample(index, curSample, slice_number, slice_size);
        % iSample位置映射到新表中两个位置
        startIndex = (iSample - 1) * slice_number + 1;
        stopIndex = iSample * slice_number;
        data_waveform(startIndex:stopIndex, :) = slices;
        data_label(startIndex:stopIndex, :) = Store_Frame_Label(iSample);
    end
    Store_Waveform = data_waveform;
    Store_Frame_Label = data_label;
    save(target, 'Store_Waveform', 'Store_Frame_Label');
end

function result = get_slice_from_one_sample(index, sample, number_sample, sample_length)
    result = zeros(number_sample, sample_length);
    for i = 1:number_sample
        left = index(i, 1);
        right = index(i, 2);
        result(i, :) = sample(left : right);
    end
end
