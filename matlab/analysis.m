close all;
clc;
clear;

% locList = {'P1', 'P2', 'P3', 'P4'};
% root = ['D:', '/WIFI_Dataset/20200520_WiFi_Data_Waveform-2'];
% records = zeros(1, 100);
% for i = 1:length(locList)
%     curP = char(locList(i));
%     namelist = dir([root, '/', curP, '/*']);
%     count = length(namelist);
%     for j = 1: count
%         file_list{j} = namelist(j).name;
%     end
%     for j = 3: count
%         addr = [root, '/', curP, '/', file_list{j}];
%         disp(addr);
%         load(addr);
%         Label = Store_Frame_Label;
%         for k = 1:length(Label)
%             pos = Label(k);
%             records(pos) = records(pos) + 1;
%         end
%     end
% end

% load('D:\WIFI_Dataset\AugData\Debug_After_Augmentation\P1\Model-B_C-1_P1_1.mat');
% Store_Frame_Label_1 = Store_Frame_Label;
% Store_Waveform_1 = Store_Waveform;
% load('D:\WIFI_Dataset\AugData\Debug_After_Augmentation\P1\Model-B_C-1_P1_2.mat');
% Store_Frame_Label_2 = Store_Frame_Label;
% Store_Waveform_2 = Store_Waveform;
% load('D:\WIFI_Dataset\AugData\Debug_Remote\Model-B_C-1_P1.mat');
% Store_Frame_Label_Remote = Store_Frame_Label;
% Store_Waveform_Remote = Store_Waveform;
% load('D:\WIFI_Dataset\AugData\Debug_Augmentation\P1\P1.mat');
% 
% tgnChannel = wlanTGnChannel('RandomStream', 'mt19937ar with seed', ...
%             'Seed', 1, 'DelayProfile', 'Model-B');
% A = Store_Waveform(1:2, :);
% A = transpose(A);
% r1 = tgnChannel(A(:, 1));
% r2 = tgnChannel(A(:, 2));
% 
% raw = A(:, 1);
% aug = r1;

load('raw.mat');
load('aug.mat');
tgnChannel = wlanTGnChannel('RandomStream', 'mt19937ar with seed', ...
            'Seed', 1, 'DelayProfile', 'Model-B');
output = tgnChannel(raw);
disp(output(20:30));
disp(tgnChannel);