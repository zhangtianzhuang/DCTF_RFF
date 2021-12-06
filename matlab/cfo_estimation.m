clc;
close all;
clear;

Disk = 'D:';
dataSource = [Disk, '/WIFI_Dataset/AugData'];
position = {'P1', 'P2', 'P3', 'P4'};
frameHeader = 24;

cfoExpectionRecord = zeros(4, 10);
cfoVarianceRecord = zeros(4, 10);
% aggregation
for iPos = 1:length(position)
    curPosition = char(position(iPos));
    curDir = [dataSource, '/CFO_Estimation_Result/', curPosition];
    fileList = getFileNameFromDir(curDir);
    for iFile = 1:length(fileList)
        load([curDir, '/', fileList{iFile}]);
        curLabel = Store_Frame_Label(1);
        CFOTable = Store_Waveform_CFO(:,1:80);
        cfoExpection = mean(mean(CFOTable));
        cfoVariance = var(var(CFOTable));
        cfoExpectionRecord(iPos, curLabel) = cfoExpection;
        cfoVarianceRecord(iPos, curLabel) = cfoVariance;
        disp(['Position: ', curPosition, ', Device: ', num2str(curLabel), ...
            ', Expection:', num2str(cfoExpection), ', Variance:', num2str(cfoVariance)]);
    end
end


% calculate cfo
% for iPos = 1:length(position)
%     curPosition = char(position(iPos));
%     curDir = [dataSource, '/CFO_Estimation/', curPosition];
%     fileList = getFileNameFromDir(curDir);
%     for iFile = 1:length(fileList)
%         load([curDir, '/', fileList{iFile}]);
%         fileNum = length(Store_Frame_Label);
%         Store_Waveform_CFO = zeros(fileNum, 82);
%         curLabel = -1;
%         for jLabel = 1:length(Store_Frame_Label)
%             if curLabel == -1
%                 curLabel = Store_Frame_Label(1);
%             end
%             signal = Store_Waveform(jLabel, frameHeader:frameHeader+80*80-1);
%             reshapeSignal = reshape(signal, 80, 80);
%             signalCFO = calculateCFO(reshapeSignal);
%             Store_Waveform_CFO(jLabel, 1:80) = signalCFO;
%             Store_Waveform_CFO(jLabel, 81) = mean(signalCFO);
%             Store_Waveform_CFO(jLabel, 82) = var(signalCFO);
%         end
%         % saved location
%         saveDirName = [dataSource, '/CFO_Estimation_Result/', curPosition];
%         mkdirIfNotExist(saveDirName);
%         saveFileName = ['WaveformCFO_', curPosition, '_D', num2str(curLabel),...
%             '.mat'];
%         save([saveDirName, '/', saveFileName], 'Store_Waveform_CFO', ...
%             'Store_Frame_Label');
%         disp(['Device ', num2str(curLabel), ' has been Finished!']);
%     end
% end

% generate data by device label
% for iPos = 1:length(position)
%     curPosition = char(position(iPos));
%     curDir = [dataSource, '/WiFi_Waveform/', curPosition];
%     fileList = getFileNameFromDir(curDir);
%     deviceNumber = countNumberOfDevice(curDir);
%     for iLabel = 1:10
%         Temp_Waveform = zeros(deviceNumber(iLabel), 8020);
%         Temp_Label = zeros(deviceNumber(iLabel), 1);
%         index = 1;
%         for iFile = 1:length(fileList)
%             load([curDir, '/', fileList{iFile}]);
%             for jLabel = 1:length(Store_Frame_Label)
%                 if Store_Frame_Label(jLabel) == iLabel
%                     Temp_Waveform(index, :) = Store_Waveform(jLabel, :);
%                     Temp_Label(index) = iLabel;
%                     index = index + 1;
%                 end
%             end
%         end
%         Store_Waveform = Temp_Waveform;
%         Store_Frame_Label = Temp_Label;
%         saveDirName = [dataSource, '/CFO_Estimation/', curPosition];
%         mkdirIfNotExist(saveDirName);
%         saveFileName = ['Waveform_', curPosition, '_D', num2str(iLabel),...
%             '.mat'];
%         save([saveDirName, '/', saveFileName], 'Store_Waveform', 'Store_Frame_Label');
%         disp(['Device ', num2str(iLabel), 'has been Finished!']);
%     end
% end

function cfo = calculateCFO(signal)
    % Input: 
    %    signal:a matrix, each of column vector is one OFDM symbol
    %    the length of a OFDM symbol is 80
    % Output:
    %    carrier frequency offset of all symbols, a row vector
    [~, n] = size(signal);
    cfo = zeros(1, n);
    for i = 1:n
        symbol = signal(:, i);
        preamble = transpose(symbol(1:16));
        cp = transpose(symbol(65:80));
        cfoEstimation = angle(preamble * cp') / -2 / pi;
        cfo(i) = cfoEstimation;
    end
end


function deviceNumber = countNumberOfDevice(path)
    deviceNumber = zeros(10, 1);
    fileList = dir([path, '/*']);
    fileNumber = length(fileList);
    fileNames = cell(fileNumber, 1);
    for i = 1:fileNumber
        fileNames{i} = fileList(i).name;
    end
    for i = 3:fileNumber
        curFileName = [path, '/', fileNames{i}];
        load(curFileName);
        for j = 1:length(Store_Frame_Label)
            label = Store_Frame_Label(j);
            if label > 10
                continue;
            end
            deviceNumber(label) = deviceNumber(label) + 1;
        end
    end
end

function fileNameList = getFileNameFromDir(path)
    fileList = dir([path, '/*']);
    fileNumber = length(fileList);
    fileNames = cell(fileNumber, 1);
    for i = 1:fileNumber
        fileNames{i} = fileList(i).name;
    end
    fileNameList = fileNames(3:end);
end

% frame_header = 24;
% Disk = 'D:';
% % Disk = '/mnt/DiskA-1.7T/ztz';
% dataSource = [Disk, '/WIFI_Dataset/AugData', '/WiFi_Waveform'];
% localList = {'P4'};
% 
% deviceCount = 11;
% countPerDevice = 200;
% for loop1 = 1:length(localList)
% %     cfoTable = zeros(deviceCount, 1000);
% %     cfoTable(:, 1) = 1;
%     FilePoint = char(localList(loop1));
%     count = zeros(deviceCount, 1);
%     index = 1;
%     Temp_Store_Frame_Label = zeros(countPerDevice * deviceCount, 1);
%     Temp_Store_Waveform = zeros(countPerDevice * deviceCount, 8020);
%     namelist = dir([dataSource, '/', FilePoint, '/*']);
%     fileNums = length(namelist);
%     for i=1:fileNums
%         file_name{i} = namelist(i).name;
%     end
%     for i = 3:fileNums
%         DataRoot = [dataSource, '/', FilePoint, '/', file_name{i}];
%         load(DataRoot);
%         for frame = 1: length(Store_Frame_Label)
%             frame_label = Store_Frame_Label(frame);
%             if frame_label == 69
%                 frame_label = 11;
%             end
%             currentCount = count(frame_label);
%             if currentCount >= 200
%                 continue;
%             end
%             Temp_Store_Waveform(index, :) = Store_Waveform(frame, :);
%             Temp_Store_Frame_Label(index,:) = frame_label;
%             count(frame_label) = count(frame_label) + 1;
%             index = index + 1;
% %             T = Store_Waveform(frame, :);
% %             T = transpose(T); % 转为列向量
% %             T = T/sqrt(mean(abs(T).^2));
% %             T = T(frame_header:end);  % 从帧头开始
% %             T1 = T(1: 160); % 短导频
% %             %% 频偏估计
% %             shortSymbolPre = transpose(T1(1:16));
% %             shortSymbolPost = transpose(T1(65:80));
% %             cfoEst = angle(shortSymbolPre * shortSymbolPost') * 64 / (2 * pi);
% %             index = cfoTable(frame_label, 1);
% %             cfoTable(frame_label, 1) = index + 1;
% %             cfoTable(frame_label, index+1) = cfoEst;
% %             disp(['file: ', file_name{i}, ', label: ',...
%               num2str(frame_label), ...
%               ', CFO: ', num2str(cfoEst)]);
%         end
%     end
%     Store_Waveform = Temp_Store_Waveform;
%     Store_Frame_Label = Temp_Store_Frame_Label;
%     save([FilePoint, '.mat'], 'Store_Waveform', 'Store_Frame_Label');
% %     save('cfoTableP4.mat', 'cfoTable');
% %     disp(count);
% %     disp(sum(count));
% %     disp(min(count));
% %     disp(max(count));
% end


% Cfo = {'cfoTableP1.mat'}; % , 'cfoTableP2.mat', 'cfoTableP3.mat', 'cfoTableP4.mat'
% for c = 1:length(Cfo)
%     load(char(Cfo(c)));
%     P1 = cfoTable;
%     for i = 1:10
%         len = P1(i, 1);
%         x = 1:len-1;
%         y = P1(i, 2:len);
% %         figure;
% %         plot(x, y);
%         disp(['location: ', char(Cfo(c)) , ', Device: ', num2str(i), '  ==>  exp:', ... 
%             num2str(mean(y)), ', var:', num2str(var(y))]);
% %         disp(num2str(mean(y)));
%     end
%     disp('-----------------------------------------------');
% end