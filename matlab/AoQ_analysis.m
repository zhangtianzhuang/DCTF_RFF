clc;
clear;
close all;

X1 = transpose(STS_LTS_Generator('LONG-F'));

respTable = zeros(64, 10);
for d = 1:10
    load(['D:\WIFI_Dataset\AugData\ClearedDataset-1-RawSlice\PointAndDevice\P2_D', num2str(d), '.mat']);
    signal = transpose(Store_Waveform(1, 24:end));
    resp = get_channel_frequency_response(signal, X1);
    respTable(:, d) = resp;
end

figure;
grid on; box on;
hold on; 

for d = 1:5
    plot(abs(respTable(:, d)), 'lineWidth', 1.5);
end
xlabel('Subcarrier');
ylabel('Frequency response');
legend('device-1', 'device-2', 'device-3', 'device-4', 'device-5');
% legend('device-6', 'device-7', 'device-8', 'device-9', 'device-10');

function resp = get_channel_frequency_response(signal, standard)
% Input：signal必须是列向量，standard：标准的长训练符号频率响应，列向量64*1
% Output：resp，信道的频率响应，列向量
    T = signal;
    T = T/sqrt(mean(abs(T).^2)); % 功率归一化
    T2 = T(161: 320);
    T2_1 = T2(33:96);
    Y1 = fftshift(fft(T2_1));
    resp = Y1 ./ standard;
end