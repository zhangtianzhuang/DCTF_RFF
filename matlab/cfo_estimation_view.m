clc;
close all;
clear;

load('cfoExpectionRecord.mat');
[m, n] = size(cfoExpectionRecord);
figure;
grid on;
box on;
xlabel('Position');
ylabel('Carrier Frequency Offset');
Marker = {'o', '+', '*', 'x', 's', ...
          '^', '>', '<', 'p', 'h'};
xlim([0.8 5.5]);
for iPosition = 1:n
    x = 1:m;
    y = cfoExpectionRecord(:, iPosition);
    hold on;
    plot(x, y, '-', 'Marker', char(Marker{iPosition}));
end
legend('Device 1', 'Device 2', 'Device 3', 'Device 4', 'Device 5', ...
    'Device 6', 'Device 7', 'Device 8', 'Device 9', 'Device 10', ...
    'location', 'SouthEast');

figure;
grid on;
box on;
xlabel('Device Label');
ylabel('Carrier Frequency Offset');
for iPosition = 1:m
    x = 1:n;
    y = cfoExpectionRecord(iPosition, :);
    hold on;
    plot(x, y, '-', 'Marker', 'o');
end
legend('Position 1', 'Position 2', 'Position 3', 'Position 4', ...
    'location', 'SouthEast');
