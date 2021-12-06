clc;
close all;
clear;

load('A.mat');
[m, n] = size(A);
source_dir = 'D:\WIFI_Dataset\AugData\ClearedDataset-20\P2';
Temp_Waveform = zeros(1000, 8020);
Temp_Frame_Label = zeros(1000, 1);
index = 1;
for row = 1:m
    s = A(row, :);
    num = s(2);
    file_name = [source_dir, '\', 'P2-', num2str(s(1))];
    load(file_name);
    j = 3;
    while j < num * 2 + 2
        pos = s(j);
        label = s(j+1);
        Temp_Waveform(index, :) = Store_Waveform(pos, :);
        Temp_Frame_Label(index) = Store_Frame_Label(pos);
        index = index + 1;
        j = j + 2;
    end
end
Store_Waveform = Temp_Waveform;
Store_Frame_Label = Temp_Frame_Label;
save('P2.mat', 'Store_Waveform', 'Store_Frame_Label');
