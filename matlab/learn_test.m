close all
clc;
clear;

load('test.mat');


% tgnChannel = wlanTGnChannel('RandomStream', 'mt19937ar with seed', ...
%             'Seed', 1, 'DelayProfile', 'Model-B');
% tgnChannel2 = wlanTGnChannel('RandomStream', 'mt19937ar with seed', ...
%             'Seed', 2, 'DelayProfile', 'Model-B');
% tgnChannel3 = wlanTGnChannel('RandomStream', 'mt19937ar with seed', ...
%             'Seed', 3, 'DelayProfile', 'Model-B');
% load('D:\WIFI_Dataset\AugData\ClearedDataset-20\P1\P1-1.mat');
% s1 = Store_Waveform(1, :);
% load('D:\WIFI_Dataset\AugData\ClearedDataset-20\P2\P2-1.mat');
% s2 = Store_Waveform(1, :);
% s1 = transpose(s1);
% s2 = transpose(s2);
% s2 = tgnChannel3(s2);
% s2(1:1000) = s2(7:1006);
% s3 = tgnChannel(s1);
% s3(1:1000) = s3(7:1006);
% s4 = tgnChannel2(s1);
% s4(1:1000) = s4(7:1006);
%% build signal
x = get_data1();
%% get frequecy response
resp = fftshift(fft(x(193:256, :)));
%% canvas
% LineWidth = 1.5;
% fig = figure; 
% hold on;
% plot(abs(resp(:,1)), 'Color', '#006dbb', 'LineStyle', '-', 'LineWidth', LineWidth);
% plot(abs(resp(:,2)), 'Color', '#ff0000', 'LineStyle', '--', 'LineWidth', LineWidth);
% plot(abs(resp(:,3)), 'Color', '#7E2F8E', 'LineStyle', '-.', 'LineWidth', LineWidth);
% plot(abs(resp(:,4)), 'Color', '#9d0723', 'LineStyle', ':', 'LineWidth', LineWidth);
% 
% % title('Frequency Response of Received Signal');
% xlabel('Subcarrier'); ylabel('Amplitude');
% xlim([1 64]);
% ylim([0 2]);
% legend('Location 1','Location 2','Location 1 w/ channel augmentation',...
%     'Location 2 w/ channel augmentation', 'location','northwest');
% xticks([1 33 64]);
% xticklabels({'-32','0','31'});
% grid on;
% box on;
% 
% function data1 = get_data1()
% tgnChannel1 = wlanTGnChannel('RandomStream', 'mt19937ar with seed', ...
%             'Seed', 1, 'DelayProfile', 'Model-D');
% tgnChannel2 = wlanTGnChannel('RandomStream', 'mt19937ar with seed', ...
%             'Seed', 1, 'DelayProfile', 'Model-E');
% % tgnChannel3 = wlanTGnChannel('RandomStream', 'mt19937ar with seed', ...
% %             'Seed', 1, 'DelayProfile', 'Model-D');
% load('D:\WIFI_Dataset\AugData\ClearedDataset-20\P1\P1-1.mat');
% s1 = transpose(Store_Waveform(1, :));
% load('D:\WIFI_Dataset\AugData\ClearedDataset-20\P2\P2-1.mat');
% s2 = transpose(Store_Waveform(10, :));
% 
% s3 = tgnChannel1(s1);
% s3(1:1000) = s3(7:1006);
% 
% s4 = tgnChannel2(s2);
% s4(1:1000) = s4(7:1006);
% data1 = [s1, s2, s3, s4];
% end


% function resp = get_frequecy_response(x)
%     % if x is a matrix, a signal must be column vector
%     resp = fftshift(fft(x(193:256, :)));
% end


% command line is redirected to files
% diary('C:\Users\ztz\Desktop\test.txt');
% diary on;
% disp('hello');
% disp('hworo');
% diary off;
% disp('hworo');

% global m;
% m = zeros(3, 3);
% disp(m);
% globalPara();
% disp(m);




% load('F:\WIFI_Dataset\20200520_WiFi_Data_Waveform_test\P1\160212-003752.mat');
% T = Store_Waveform(2, :);
% T = Store_STF_Equ(1, :);
% figure;
% plot(abs(T));

% a = [1, 1, 1, 1;
%     1, 1, 1, 1;
%     1, 1, 1, 1;
%     1, 1, 1, 1
%     ];
% [x1, y1, x2, y2] = smooth_image(a);

% res = build_diff_str_name([1,20,30]);

% c = get_filename('D:\WIFI_Dataset\20200520_WiFi_Data_Waveform\P1\*');
% global LONG_FREQUENCY_RESP;
% global deviceIndex;
% global N;
% global Plot_Table_Size;
% global Plot_Table_Max;
% global Process_Data_Offset;
% global B;
% global TargetPath;

% deviceIndex = zeros(10, 1);
% LONG_FREQUENCY_RESP = transpose(STS_LTS_Generator('LONG-F'));
% N = 90;
% B = 320;
% Plot_Table_Size = 31;  % DCTF Size
% Plot_Table_Max = 1.4;  % DCTF Size
% Process_Data_Offset = 0; % Delete some samples for generating DCTF
% TargetPath = 'F:\WIFI_Dataset\20200520_WiFi_Data_Waveform_test\test';
% channel = 1;
% signalToImg(T, 1, 'no', 'P1', 1, '1');
