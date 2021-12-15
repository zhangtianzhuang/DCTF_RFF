clc;
close all;
clear;

figure;
hold on;
x = [5 10 15 20 25 30 35];
y1 = [71.3889, 89.4444, 93.0556, 95.2778, 97.2222, 97.5, 98.6111];
y2 = [90.975, 97.125, 98.9, 98.85, 98.975, 99.15, 98.925];
plot(x, y1, 'r-', 'Marker', 'o', 'MarkerFaceColor', 'r', 'LineWidth', 1.5);
plot(x, y2, 'b-', 'Marker', 's', 'MarkerFaceColor', 'b', 'LineWidth', 1.5);
xlabel('SNR(dB)');
ylabel('Identification Accuracy(%)');
grid on; box on;
legend('ours', 'raw data', 'location', 'southeast');


figure;
hold on;
y3 = [72.6111, 83.8333, 86.1667, 89.4444, 92.0556, 93.1667, 94.6667];
y4 = [41.55, 42.65, 44.775, 44.325, 44.3, 44.775, 44.725];
plot(x, y3, 'r-', 'Marker', 'o', 'MarkerFaceColor', 'r', 'LineWidth', 1.5);
plot(x, y4, 'b-', 'Marker', 's', 'MarkerFaceColor', 'b', 'LineWidth', 1.5);
xlabel('SNR(dB)');
ylabel('Identification Accuracy(%)');
grid on; box on;
legend('ours', 'raw data', 'location', 'southeast');

figure;
hold on;
plot(x, y1, 'r-', 'Marker', 'o', 'MarkerFaceColor', 'r', 'LineWidth', 1.5);
plot(x, y3, 'b-', 'Marker', 's', 'MarkerFaceColor', 'b', 'LineWidth', 1.5);
xlabel('SNR(dB)');
ylabel('Identification Accuracy(%)');
grid on; box on;
legend('same channel', 'different channel', 'location', 'southeast');

