clc;
close all;
clear;

figure;
hold on;
x = [5 10 15 20 25 30 35];
y1 = [71.6111, 80.0278, 83.8611, 86.7222, 90.4167, 92.8889, 97.0833];
y2 = [71.2778, 81.1944, 84.0278, 87.0278, 89.6944, 92.5833, 96.1667];
y3 = [72.4444, 81.8333, 84.7778, 87.0, 90.0, 92.1944, 96.4722];
y4 = [72.7778, 81.7778, 84.9444, 87.4722, 90.2778, 93.1944, 96.5278];
y5 = [70.9167, 80.9444, 85.0278, 87.5556, 90.3333, 92.8611, 96.1111];
y6 = [73.4167, 81.6667, 84.5833, 87.0833, 89.8611, 92.8333, 96.1389];
plot(x, y1, 'r-', 'Marker', 'o', 'MarkerFaceColor', 'r', 'LineWidth', 1.5);
plot(x, y2, 'g-', 'Marker', '*', 'MarkerFaceColor', 'g', 'LineWidth', 1.5);
plot(x, y3, 'b-', 'Marker', 's', 'MarkerFaceColor', 'b', 'LineWidth', 1.5);
plot(x, y4, 'm-', 'Marker', 'v', 'MarkerFaceColor', 'm', 'LineWidth', 1.5);
plot(x, y5, 'k-', 'Marker', '^', 'MarkerFaceColor', 'k', 'LineWidth', 1.5);
plot(x, y6, 'c-', 'Marker', '>', 'MarkerFaceColor', 'c', 'LineWidth', 1.5);
xlabel('SNR(dB)');
ylabel('Identification Accuracy(%)');
grid on;

legend('batch size 32', 'batch size 64', 'batch size 128', ...
    'batch size 256', 'batch size 512', 'batch size 1024', ...
    'location', 'southeast');

% legend('batch size 64', 'batch size 128',... % 'Location A,B', ...
%     'location', 'southeast');
%     'Location B Train, A Test', 'Location A Train, B Test',...
    
% legend('Location A', 'Location B', 'Location A,B', ...
%     'Location B Train, A Test', 'Location A Train, B Test',...
%     'location', 'southeast');


%xlim([13 37]);
%ylim([91 98]);
