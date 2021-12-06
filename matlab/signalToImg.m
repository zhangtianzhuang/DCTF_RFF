function signalToImg(signal, label, snr, FilePoint, DiffInterval, channel)
    if label > 37
        return;
    end
    global LONG_FREQUENCY_RESP;
    global N;
    global deviceIndex;
    global dataTarget;
    T = preProcessSignal(signal, snr);
    Equ = equalizedSignal(T, LONG_FREQUENCY_RESP);
    Diff = Equ(:,1:N-DiffInterval) .* conj(Equ(:,1+DiffInterval:N));
    img = plotHeatMap(Diff);
    
    fnDir = [dataTarget, '/snr_', num2str(snr), '/', FilePoint, '/', num2str(label)];
    if exist(fnDir, 'dir') == 0
        mkdir(fnDir);
    end
    
    fnFileName = ['/D', num2str(label), ...
                    '_', FilePoint, ...
                    '_S', snr, ...
                    '_I', num2str(DiffInterval), ...
                    '_C', channel,...
                    '_No', num2str(deviceIndex(label)), ...
                    '.jpg'];
    allPath = [fnDir, fnFileName];
    imwrite(img, allPath);
    deviceIndex(label) = deviceIndex(label) + 1;
    disp([allPath, ' ', num2str(label)]);
end

function res = plotHeatMap(Diff)
    global Plot_Table_Size;
    global Plot_Table_Max;
    global Process_Data_Offset;
    % 绘制热力图
    HeatMapData = reshape(Diff, [], 1);
    Plot_Table = F_Get_Data_Table(Plot_Table_Size,Plot_Table_Max, ...
        HeatMapData, Process_Data_Offset);
    % 消除中心点值引入的影响
    for x1=Plot_Table_Size-2:Plot_Table_Size+2
        for x2=Plot_Table_Size-2:Plot_Table_Size+2
            Plot_Table(x1,x2) = 0;
        end
    end
    min_table = min(min(Plot_Table));  % minmum of the matrix
    max_table = max(max(Plot_Table));  % maxmum of the matrix
    % 归一化
    Plot_Table = Plot_Table ./ (max_table-min_table) * 255;
    % 消除边缘节点的影响
    Plot_Table(1:2,:) = 0;
    Plot_Table(end-1:end, :) = 0;
    Plot_Table(:,1:2) = 0;
    Plot_Table(:,end-1:end) = 0;
    res = graycolor(Plot_Table);
end

function res = preProcessSignal(signal, snr)
    [m, ~] = size(signal);
    T = signal(24: end);
    if m == 1
        T = transpose(T);
    end
    res = T/sqrt(mean(abs(T).^2));
    % 添加加性高斯白噪声
    if strcmp(snr, 'no') == 0
        s = str2num(snr);
        res = awgn(T, s, 'measured');
    end
end

function res = equalizedSignal(signal, X)
    %     T1 = T(1: 160); % 短导频
    global N;
    global B;
    T2 = signal(161: 320); % 长导频
    D = zeros(64, N);
    for k=1:N
        left = B + (k-1) * 80 + 17;  right = B + k * 80;
        D(:,k) = signal(left:right);
    end
    T2_1 = T2(33:96); % 第1个长符号
%     T2_2 = T2(97:160); % 第2个长符号
    Y1 = fftshift(fft(T2_1)); % 第1个长符号频率响应
    H = Y1 ./ X; % 信道估计
    res = fftshift(fft(D),1) ./ H; % 均衡信号
end