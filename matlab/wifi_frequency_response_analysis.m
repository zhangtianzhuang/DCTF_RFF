clc;
clear;
close all;

load(['D:\WIFI_Dataset\AugData\ClearedDataset-1-RawSlice\', ...
    'PointAndDevice\P2_D', num2str(4), '.mat']); 
figure;
grid on; box on;
hold on;
for i =1:10
    signal = transpose(Store_Waveform(i, 24:end));
    aoq = get_aoq(signal);
%     x = 1:length(aoq);
    plot(abs(aoq), 'k-', 'lineWidth', 1);
end
% xlabel('Subcarrier');
% ylabel('Frequency response');
% legend('device-1', 'device-2', 'device-3', 'device-4', 'device-5');

function aoq = get_aoq(signal)
% Input：signal必须是列向量，standard：标准的长训练符号频率响应，列向量64*1
% Output：resp，信道的频率响应，列向量
    T = signal;
%     T = T/sqrt(mean(abs(T).^2)); % 功率归一化
    T2 = T(161: 320); % 2个长训练符号
    T2_1 = T2(33:96); % 第1个长训练符号
    T2_2 = T2(97:160); % 第2个长训练符号

    F1 = fftshift(fft(T2_1));
    F2 = fftshift(fft(T2_2));
    % 去掉前6个和后5个
    F1 = F1(7:end-5);
    F2 = F2(7:end-5);
    % 去掉中间的节点
%     F1 = [F1(1:26); F1(28:end)];
%     F2 = [F2(1:26); F2(28:end)];
    aoq = F1 ./ F2;
end

