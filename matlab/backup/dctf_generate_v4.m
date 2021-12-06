% 增加不同的差分间隔，并绘制在相同的图片中

clc;
close all;
clear;

%% 数据集
% 东南大学数据集
Disk = 'F:';
IS_PLOT = 0;
DataRoot = [Disk, '/WIFI_Dataset/20200520_WiFi_Data_Waveform'];
X1 = STS_LTS_Generator('LONG-F');
X1 = transpose(X1);
% add_snr = '10';
% {'0','5','10','15','20', '25', '30', 'no'};
snrList =  {'no'}; % add_snr = 'no' 表示不添加噪声
locList = {'P1', 'P2','P3', 'P4'}; %
for loop1 = 1:length(locList)
    FilePoint = char(locList(loop1)); % FilePoint = 'P4';   % 选择地点
    disp(['Location: ' FilePoint] );
    for loop2 = 1:length(snrList)
        add_snr = char(snrList(loop2));
        disp(['SNR: ' add_snr] );

        ImageTargetPath = [DataRoot, '/DCTF_IMG/snr_', add_snr];
        namelist = dir([DataRoot,'/', FilePoint, '/*']);
        fileNums = length(namelist);
        DiffInterval = 1;

        for i=1:fileNums
            file_name{i} = namelist(i).name;
        end
        fileIndex = ones(37, 1);
        frame_header = 24; % 帧头开始位置
        % i从3开始，1和2是隐藏目录，分别是.和..
        is_test = 0;
        if is_test == 1
            fileNums = 3;
        end    

        for i=3:fileNums
            load([DataRoot, '/', FilePoint, '/', file_name{i}]);
            frameNums = length(Store_Frame_Label);
            if is_test == 1
                frameNums = 1;
            end
            for frame = 1:frameNums
                frame_label = Store_Frame_Label(frame);
                if frame_label > 37
                    continue;
                end
                T = Store_Waveform(frame, :);
                T = transpose(T); % 转为列向量
                T = T/sqrt(mean(abs(T).^2));
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
                if IS_PLOT == 1
                    fig = figure(1);
                    plot(Diff, 'k.');
                    grid;
                    axis([-3.5,3.5,  -3.5,3.5]);
                end
                % 将标准为69的设备修改为11
                newFrameLabel = frame_label;
%                 if newFrameLabel == 69
%                     newFrameLabel = 11;
%                 end
                % 保存当前图片
                fnDir = [ImageTargetPath, '/', FilePoint, '/', num2str(newFrameLabel)];
                if exist(fnDir, 'dir') == 0
                    mkdir(fnDir);
                end
                % writeFig2Jpg(fig, fn);
                % 绘制热力图
                HeatMapData = reshape(Diff, [],1);
                Plot_Table_Size = 31;  % DCTF Size
                Plot_Table_Max = 1.4;  % DCTF Size
                Process_Data_Offset = 0; % Delete some samples for generating DCTF
                Plot_Table = F_Get_Data_Table(Plot_Table_Size,Plot_Table_Max, ...
                    HeatMapData, Process_Data_Offset);
                % 消除中心点值引入的影响
                center = Plot_Table_Size;
                for x1=center-2:center+2
                    for x2=center-2:center+2
                        Plot_Table(x1,x2) = 0;
                    end
                end
                min_table=min(min(Plot_Table));  % minmum of the matrix
                max_table=max(max(Plot_Table));  % maxmum of the matrix
                % 归一化
                Plot_Table = Plot_Table ./ (max_table-min_table) * 255;
                % 消除边缘节点的影响
                Plot_Table(1:2,:) = 0;
                Plot_Table(end-1:end,:) = 0;
                Plot_Table(:,1:2) = 0;
                Plot_Table(:,end-1:end) = 0;
                
%                 [a1, b1, a2, b2] = smooth_image(Plot_Table);
%                 Plot_Table(:,:) = 0;
%                 Plot_Table(a1-3:a1+3, b1-3:b1+3) = 200;
%                 Plot_Table(a2-3:a2+3, b2-3:b2+3) = 200;
                
                img = graycolor(Plot_Table);
                imshow(img);
                DCTF_fnFile = ['/D', num2str(newFrameLabel), ...
                    '_', FilePoint, ...
                    '_S', add_snr, ...
                    '_I', num2str(DiffInterval),... 
                    '_No', num2str(fileIndex(newFrameLabel)),...
                    '.jpg'];
                DCTF_fn = [fnDir, DCTF_fnFile];
                imwrite(img, DCTF_fn);
                fileIndex(newFrameLabel) = fileIndex(newFrameLabel) + 1;
                disp([DCTF_fn, ' ', num2str(newFrameLabel)]);
            end
        end
    end
end