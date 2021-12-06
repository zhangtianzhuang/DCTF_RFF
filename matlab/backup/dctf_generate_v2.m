clc;
close all;
clear;

%% 数据集
% 东南大学数据集
Disk = 'D:';
IS_PLOT = 0;
DataRoot = [Disk,'\WIFI_Dataset\20200520_WiFi_Data_Waveform'];
FilePoint = 'P1';
add_snr = 'no';
ImageTargetPath = [Disk, '\WIFI_Dataset\20200520_WiFi_Data_Waveform',...
                    '\DCTF_IMG\SNR_', add_snr];
namelist = dir([DataRoot,'\', FilePoint, '\', '*']);
fileNums = length(namelist);
DiffInterval = 1;
diary_path = [ImageTargetPath, '\', FilePoint];
if exist(diary_path, 'dir') == 0
    mkdir(diary_path);
end
diary([diary_path, '\', num2str(DiffInterval), '.txt']);
diary on;
for i=1:fileNums
    file_name{i} = namelist(i).name;
    disp([DataRoot, '\', FilePoint, '\', file_name{i}]);
end
fileIndex = ones(11, 1);
frame_header = 24; % 帧头开始位置
% i从3开始，1和2是隐藏目录，分别是.和..
is_test = 0;
if is_test == 1
    fileNums = 3;
end
for i=3:fileNums
    load([DataRoot, '\', FilePoint, '\', file_name{i}]);
    frameNums = length(Store_Frame_Label);
    if is_test == 1
        frameNums = 1;
    end
    for frame = 1:frameNums
        frame_label = Store_Frame_Label(frame);
        T = Store_Waveform(frame, :);
        T = transpose(T); % 转为列向量
        
        % 添加加性高斯白噪声
        if strcmp(add_snr, 'no') == 0
            T = awgn(T, str2num(add_snr), 'measured');
        end
        
        % 当前帧所有数据
        if IS_PLOT == 1
        figure; plot(abs(T)); title('All Data.');
        end
        T = T(frame_header:end);  % 从帧头开始
        T1 = T(1: 160); % 短导频
        T2 = T(161: 320); % 长导频
        % 验证CP
        if IS_PLOT == 1
        figure; plot(abs(T2(1:end))); hold on;
        plot(abs(T2(129: 160)), 'r'); title('验证CP');
        end
        %%%%%%%%%%%%%%%% 验证帧头合理性，在这打断点 %%%%%%%%%%%%%%%% 
        % 数据符号
        B = 320;
        N = 90; % capture OFDM符号的数量
        D = zeros(64, N);
        for k=1:N
            left = B + (k-1) * 80 + 17;  right = B + k * 80;
            D(:,k) = T(left:right);
        end
        T2_1 = T2(33:96);
        T2_2 = T2(97:160);
        % 绘制第1个长符号和第2个长符号
        if IS_PLOT == 1
        figure;
        subplot(2,1,1);
        plot(abs(T2_1));
        subplot(2,1,2);
        plot(abs(T2_2));
        end
        % 接收的第1个长符号的频率响应
        Y1 = fftshift(fft(T2_1));
        
        if IS_PLOT == 1
        figure; plot(abs(Y1), 'k-', 'LineWidth', 2);
        title('Frequency Response of Received Signal');
        xlabel('Frequency'); ylabel('Amplitude');
        end

        % 标准的长符好频率响应
        X1 = STS_LTS_Generator('LONG-F');
        X1 = transpose(X1);
        % x1 = STS_LTS_Generator('LONG-T');
        if IS_PLOT == 1
        figure; 
        plot(abs(X1), 'k-', 'LineWidth', 2);
        title('Standard OFDM Frequency Response');
        xlabel('Frequency');ylabel('Amplitude');
        end
        % 信道估计
        H = Y1 ./ X1;
        if IS_PLOT == 1
        figure; 
        plot(abs(H), 'k-', 'LineWidth', 2);
        xlabel('Frequency');ylabel('Amplitude');
        title('信道H');
        end
        % 时域信号转为频域
        E2 = fftshift(fft(D),1) ./ H;
        if IS_PLOT == 1
        figure; plot(E2, 'r.');
        end
        % 差分
        Diff = E2(:,1:N-DiffInterval) .* conj(E2(:,1+DiffInterval:N));
        fig = figure(1);
        plot(Diff, 'k.');
        grid;
        axis([-3.5,3.5,  -3.5,3.5]);
        % 保存当前图片
        fnDir = [ImageTargetPath, '\', FilePoint, '\', ...
            num2str(frame_label),... 
            '\', 'DiffInterval_', num2str(DiffInterval)];
        
        filePoint = frame_label;
        if filePoint == 69 
            filePoint = 11;
        end
        
        fnFile = ['\', num2str(fileIndex(filePoint)), '.jpg'];
        fn = [fnDir, fnFile]; % 存储路径
        if exist(fnDir, 'dir') == 0
            mkdir(fnDir);
        end
        
        % 绘制热力图
        HeatMapData = reshape(Diff, [],1);
        Plot_Table_Size = 31;  % DCTF Size
        Plot_Table_Max = 1.4;  % DCTF Size 
        Process_Data_Offset = 0; % Delete some samples for generating DCTF
        Plot_Table = F_Get_Data_Table(Plot_Table_Size,Plot_Table_Max, ...
            HeatMapData, Process_Data_Offset);
        % 消除中心点值引入的影响
        center = Plot_Table_Size;
        for x1=center-5:center+5
            for x2=center-5:center+5
                Plot_Table(x1,x2) = 0;
            end
        end
        min_table=min(min(Plot_Table));  % minmum of the matrix
        max_table=max(max(Plot_Table));  % maxmum of the matrix
        % 归一化
        Plot_Table=Plot_Table./(max_table-min_table)*255;
        img = graycolor(Plot_Table);

        DCTF_fnFile = ['\dctf_', num2str(fileIndex(filePoint)), '.jpg'];
        DCTF_fn = [fnDir, DCTF_fnFile];
        
        imwrite(img, DCTF_fn);
        fileIndex(filePoint) = fileIndex(filePoint) + 1;
        disp([DCTF_fn, ', diff=', num2str(DiffInterval)]);
    end
end

