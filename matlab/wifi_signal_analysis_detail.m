clc;
close all;
clear;

%% 
X1 = STS_LTS_Generator('LONG-F');
X1 = transpose(X1);
%% 
IS_PLOT = 1;
DiffInterval = 1;
frame_header = 24;
DataRoot = 'D:\WIFI_Dataset\AugData\P1WithLittle2TrainTest\P2\Model-B_C-21_Model-B_C-11_P2-1.mat';
load(DataRoot);
frame = 2;
frame_label = Store_Frame_Label(frame);

T = Store_Waveform(:, frame);
T = transpose(T); 
T = T/sqrt(mean(abs(T).^2));

if IS_PLOT == 1
    figure; plot(abs(T)); title('All Data.');
end
T = T(frame_header:end);
T1 = T(1: 160); 
T2 = T(161: 320);

%% 
shortSymbolPre = transpose(T1(1:16));
shortSymbolPost = transpose(T1(65:80));
cfoEst = angle(shortSymbolPre * shortSymbolPost') * 64 / (2 * pi);
disp(cfoEst);
%%
if IS_PLOT == 1
    figure; plot(abs(T2(1:end))); hold on;
    plot(abs(T2(129: 160)), 'r'); title('验证CP');
end
%% 
B = 320;
N = 90; 
D = zeros(64, N);
for k=1:N
    left = B + (k-1) * 80 + 17;  right = B + k * 80;
    D(:,k) = T(left:right);
end
T2_1 = T2(33:96);
T2_2 = T2(97:160);
%% 
if IS_PLOT == 1
    figure;
    subplot(2,1,1);
    plot(abs(T2_1));
    subplot(2,1,2);
    plot(abs(T2_2));
end
%% 
Y1 = fftshift(fft(T2_1));

if IS_PLOT == 1
    figure; plot(abs(Y1), 'k-', 'LineWidth', 2);
    title('Frequency Response of Received Signal');
    xlabel('Frequency'); ylabel('Amplitude');
end

%% 
H = Y1 ./ X1;
if IS_PLOT == 1
    figure;
    plot(abs(H), 'k-', 'LineWidth', 2);
    xlabel('Frequency');ylabel('Amplitude');
    title('信道H');
end
%% 
E2 = fftshift(fft(D),1) ./ H;
if IS_PLOT == 1
    figure; plot(E2, 'r.');
end
%% 
Diff = E2(:,1:N-DiffInterval) .* conj(E2(:,1+DiffInterval:N));
if IS_PLOT == 1
    fig = figure(1);
    plot(Diff, 'k.');
    grid;
    axis([-3.5,3.5,  -3.5,3.5]);
end
newFrameLabel = frame_label;

% writeFig2Jpg(fig, fn);
%% 
HeatMapData = reshape(Diff, [],1);
Plot_Table_Size = 31;  % DCTF Size
Plot_Table_Max = 1.4;  % DCTF Size
Process_Data_Offset = 0; % Delete some samples for generating DCTF
Plot_Table = F_Get_Data_Table(Plot_Table_Size,Plot_Table_Max, ...
    HeatMapData, Process_Data_Offset);
%% 
center = Plot_Table_Size;
for x1=center-2:center+2
    for x2=center-2:center+2
        Plot_Table(x1,x2) = 0;
    end
end
min_table=min(min(Plot_Table));  % minmum of the matrix
max_table=max(max(Plot_Table));  % maxmum of the matrix
%% 
Plot_Table = Plot_Table ./ (max_table-min_table) * 255;
%% 
Plot_Table(1:2,:) = 0;
Plot_Table(end-1:end,:) = 0;
Plot_Table(:,1:2) = 0;
Plot_Table(:,end-1:end) = 0;

img = graycolor(Plot_Table);
imshow(img);
