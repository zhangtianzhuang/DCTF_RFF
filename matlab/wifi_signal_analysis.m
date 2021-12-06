clear;
close all;
clc;

load('C:\Users\ztz\Desktop\P4.mat');
T = transpose(Store_Waveform(13, :));
figure;
plot(abs(T));
tgnChannel = wlanTGnChannel('RandomStream', 'mt19937ar with seed', ...
            'Seed', 101, 'DelayProfile', 'Model-B');
T2 = tgnChannel(T);
figure;
plot(abs(T2));
% load('D:\WIFI_Dataset\WiFi_Data_Waveform_Filtered\P1\P1.mat');
% P1
% pos = [142, 152, 161,  3, 12, 22,  153, 162, 172,  5, 14, 128,  44, 54, 167, ...
%     47, 66, 85, 36, 83, 93,  34, 53, 72,  37, 46, 56, 70, 80, 89]';
% P2
% pos = [128,157,192, 22,31,41, 82,101,110, 18, 28, 46, 29, 38, 47, 54,...
%     73,92, 60, 88, 107, 57,114,152, 14, 51, 89, 21,30,188]';

% P3
% pos = [2,10,19, 175,184,194, 150,160,197, 65,73,82, 57,66,74,...
% 34,51,68,21,29,38,30,47,64,155,164,173, 166,176,195 ]';

% signal = zeros(30, 8020);
% label = zeros(30, 1);
% for i = 1:length(pos)
%     signal(i, :) = Store_Waveform(pos(i), :);
%     label(i) = Store_Frame_Label(pos(i));
% end
% Store_Waveform = signal;
% Store_Frame_Label = label;
% save('D:\WIFI_Dataset\WiFi_Data_Waveform_Filtered\P3.mat', 'Store_Frame_Label', 'Store_Waveform');

% tgnChannel = wlanTGnChannel('RandomStream', 'mt19937ar with seed', ...
%             'Seed', 400, 'DelayProfile', 'Model-A');
% channelInfo = info(tgnChannel)


% load('D:\WIFI_Dataset\20200520_WiFi_Data_Waveform\P1\160212-003752.mat');
% % Store_Waveform = transpose(Store_Waveform);
% T = Store_Waveform(200, 24:124);
% figure;
% plot(abs(T));
