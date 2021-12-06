clc;
close all;
clear;

figure;
hold on;
x = [5 10 15 20 25 30 35];
y1 = [74.694, 84.361, 87.694, 89.139, 92.528, 95.222, 98.167];
plot(x, y1, 'r-', 'Marker', 'o', 'MarkerFaceColor', 'r', 'LineWidth', 1.5);
xlabel('SNR(dB)');
ylabel('Identification Accuracy(%)');
grid on;
% legend('Location A', 'Location B', 'Location A,B', ...
%     'Location B Train, A Test', 'Location A Train, B Test', 'location', 'southeast');
%xlim([13 37]);
%ylim([91 98]);
