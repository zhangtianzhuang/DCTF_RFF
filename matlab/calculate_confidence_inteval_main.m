clear;
clc;
close all;



% load('WaveformCFO_P1_D1.mat');
% A = Store_Waveform_CFO(:,1:80);
% R = calculate_confidence_inteval(A);
% x = 1:80;
% figure;
% hold on;
% fill([x, flip(x)], [R(1,:), flip(R(3,:))], 'g', 'FaceColor', [1 0.8 0.8], 'EdgeColor', 'none');
% % plot(x, R(1,:), 'b');
% plot(x, R(2,:), 'k--');
% box on;
% grid on;

% plot(x, R(3,:), 'g');
% B = transfer_scatter(A);
% plot(B(:,1), B(:,2), 'k.');
% index = 57;
% T = Store_Waveform_CFO(:,index);
% ratio = length(find(T>R(3,index) & T < R(1,index))) / length(T)

%%
P = zeros(80, 10);
for i = 1:10
    load(['D:\WIFI_Dataset\AugData\CFO_Estimation_Result\P1\', ...
        'WaveformCFO_P1_D', num2str(i),'.mat']);
    A = Store_Waveform_CFO(:,1:80);
    P(:, i) = transpose(mean(A));
end
R = calculate_confidence_inteval(P);
x = 1:10;
fig = figure;
hold on;
fill([x, flip(x)], [R(1,:), flip(R(3,:))], 'g', 'FaceColor', [0.8863 0.9412 0.8510],...
    'EdgeColor', 'none');
plot(x, R(2,:), 'r-o');
box on;
grid on;
% 将图片保存到文件中
save_png(fig, 'C:/Users/ztz/Desktop/test.png');

%% 

function res = transfer_scatter(samples)
    [rowNum, colNum] = size(samples);
    all = rowNum * colNum;
    y = reshape(samples, [all, 1]);
    x = 1:colNum;
    x = repmat(x, [rowNum, 1]);
    x = reshape(x, [all, 1]);
    res = [x, y];
end


function res = calculate_confidence_inteval(samples)
    % 90%, z = 1.645
    % 95%, z = 1.96
    % 99%, z = 2.58
    [rowNum, colNum] = size(samples);
    confidenceInterval = zeros(3, colNum);
    n = rowNum; % the number of samples
    std_dev = std(samples); % standard deviation
    avg = mean(samples);
    interval_size = std_dev;
    confidenceInterval(1,:) = avg + interval_size*2;
    confidenceInterval(2,:) = avg;
    confidenceInterval(3,:) = avg - interval_size*2;
    res = confidenceInterval;
end