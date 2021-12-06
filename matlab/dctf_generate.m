clc;
close all;
clear;

%SystemType = 'windows';
SystemType = 'linux';

% windows config
if strcmp(SystemType, 'windows')
    dataRoot = 'D:/WIFI_Dataset/';
else
% linux config
    dataRoot = '/mnt/DiskA-1.7T/ztz/WIFI_Dataset/';
end

IS_PLOT = 0;
dataType = 'ClearedDataset-20_ModelB_DS_S501-900';
dataSource = [dataRoot, 'AugData/', dataType];
dataTarget = [dataRoot, 'DCTF_Image/', dataType];
X1 = STS_LTS_Generator('LONG-F');

X1 = transpose(X1);
% {'0', '5', '10', '15', '20', '25', '30', '35', 'no'};
snrList =  {'5', '10', '15', '20', '25', '30', 'no'};
localList = {'P1', 'P2'}; % , 'P2', 'P3', 'P4'
frame_header = 24;

for loop1 = 1:length(localList)
    FilePoint = char(localList(loop1));
    disp(['Location: ' FilePoint]);
    for loop2 = 1:length(snrList)
        add_snr = char(snrList(loop2));
        disp(['SNR: ' add_snr] );
        ImageTargetPath = [dataTarget, '/snr_', add_snr];
        SourcePath = [dataSource, '/', FilePoint, '/*'];
        namelist = dir(SourcePath);
        fileNums = length(namelist);
        DiffInterval = [1];
        for i=1:fileNums
            file_name{i} = namelist(i).name;
        end
        fileIndex = ones(37, 1);
        
        is_test = 0;
        if is_test == 1
            fileNums = 3;
        end
        % variable i starts at 3
        for i=3:fileNums
            load([dataSource, '/', FilePoint, '/', file_name{i}]);
            frameNums = length(Store_Frame_Label);
            if is_test == 1
                frameNums = 1;
            end
            for frame = 1:frameNums
                frame_label = Store_Frame_Label(frame);
                if frame_label > 37
                    continue;
                end
                if isrow(Store_Frame_Label)
                    T = Store_Waveform(:, frame);
                else
                    T = Store_Waveform(frame, :);
                    T = transpose(T);
                end

                T = T/sqrt(mean(abs(T).^2));
                % add additive white Gaussian noise
                if strcmp(add_snr, 'no') == 0
                    T = awgn(T, str2num(add_snr), 'measured');
                end

                % all data
                if IS_PLOT == 1
                    figure; plot(abs(T)); title('All Data.');
                end
                T = T(frame_header:end);  % 从帧头开始
                T1 = T(1: 160); % 短导频
                T2 = T(161: 320); % 长导频
                % CP 
                if IS_PLOT == 1
                    figure; plot(abs(T2(1:end))); hold on;
                    plot(abs(T2(129: 160)), 'r'); title('CP Check');
                end
                
                B = 320;
                N = 90; % capture OFDM symbols
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
                % Channel Estimation
                H = Y1 ./ X1;
                if IS_PLOT == 1
                    figure;
                    plot(abs(H), 'k-', 'LineWidth', 2);
                    xlabel('Frequency');ylabel('Amplitude');
                    title('Channel H');
                end
                % 时域信号转为频域
                E2 = fftshift(fft(D),1) ./ abs(H);
                if IS_PLOT == 1
                    figure; plot(E2, 'r.');
                end
                
                % 差分
                Diff = cell(length(DiffInterval), 1);
                for pos = 1:length(DiffInterval)
                    Diff{pos} = E2(:,1:N-DiffInterval(pos)) .* conj(E2(:,1+DiffInterval(pos):N));
                    if IS_PLOT == 1
                        fig = figure(1);
                        plot(Diff{pos}, 'k.');
                        grid;
                        axis([-3.5,3.5,  -3.5,3.5]);
                    end
                end
                
                newFrameLabel = frame_label;
                % 保存当前图片
                fnDir = [ImageTargetPath, '/', FilePoint, '/', num2str(newFrameLabel)];
                if exist(fnDir, 'dir') == 0
                    mkdir(fnDir);
                end

                % 绘制热力图
                Plot_Table_Size = 31;  % DCTF Size
                Plot_Table_Max = 1.4;  % DCTF Size
                Process_Data_Offset = 0; % Delete some samples for generating DCTF
                
                PlotTableSet = cell(length(DiffInterval), 1);
                for pos = 1:length(DiffInterval)
                    HeatMapData = reshape(Diff{pos}, [], 1);
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
                    PlotTableSet{pos} = Plot_Table;
                end
                
                for pos = 1:length(PlotTableSet)
                    img = graycolor(PlotTableSet{pos});
                    diff_name = num2str(DiffInterval(pos));
                    DCTF_fnFile = ['/D', num2str(newFrameLabel), ...
                        '_', file_name{i}, '_Pos', num2str(frame), ...
                        '_', FilePoint, ...
                        '_S', add_snr, ...
                        '_I', diff_name, ...
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
end
