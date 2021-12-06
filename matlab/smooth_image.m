function [avg_x1, avg_y1, avg_x2, avg_y2] = smooth_image(input)
% input 是一个方阵，如果input的维度n是偶数，则需要把中间的两行和两列合并
    [m, n] = size(input);
    if m ~= n
        ME = MException('the input must be a phalanx');
        throw(ME);
    end
    if mod(m, 2) == 1
        res = input;
    else
        mid =  m / 2;
        input(mid, :) = input(mid, :) + input(mid+1, :);
        input(mid+1:end-1, :) = input(mid+2:end, :);
        
        input(:, mid) = input(:, mid) + input(:, mid+1);
        input(:, mid+1:end-1) = input(:, mid+2:end);
        
        input = input(1:end-1, 1:end-1);
        res = input;
    end
    
    min_value = 64 * 255;
    flag = ''; % 标记采用哪个轴对图像进行划分，x,y,-1,1
    c = length(res);
    mid = ceil(c / 2);
    % 计算x轴的和
    x = sum(res(mid, :));
    if x < min_value
        min_value = x;
        flag = 'x';
    end
    x = sum(res(:, mid));
    if x < min_value
        min_value = x;
        flag = 'y';
    end
    x = trace(res);
    if x < min_value
        min_value = x;
        flag = '-1';
    end
    x = 0;
    for i = 1:c
        x = x + res(c+1-i, i);
    end
    if x < min_value
        min_value = x;
        flag = '1';
    end
%     disp(['min_value is ', num2str(min_value)]);
    
    avg_x1 = 0;
    avg_y1 = 0;
    avg_x2 = 0;
    avg_y2 = 0;
    count1 = 0;
    count2 = 0;
    if flag == 'x'
        for i = 1 : c
            for j = 1 : c
               if res(i, j) > 0 && i ~= j
                   if i < mid
                       avg_x1 = avg_x1 + res(i, j) * i;
                       avg_y1 = avg_y1 + res(i, j) * j;
                       count1 = count1 + res(i, j);
                   else
                       avg_x2 = avg_x2 + res(i, j) * i;
                       avg_y2 = avg_y2 + res(i, j) * j;
                       count2 = count2 + res(i, j);
                   end
               end
            end
        end
    elseif flag == 'y'
        for i = 1 : c
            for j = 1 : c
                if res(i, j) > 0 && i ~= j
                    if j < mid
                        avg_x1 = avg_x1 + res(i, j) * i;
                        avg_y1 = avg_y1 + res(i, j) * j;
                        count1 = count1 + res(i, j);
                    else
                       avg_x2 = avg_x2 + res(i, j) * i;
                       avg_y2 = avg_y2 + res(i, j) * j;
                       count2 = count2 + res(i, j);
                    end
                end
            end
        end    
    elseif flag == '1'
        for i = 1 : c
            for j = 1 : c
                if res(i, j) > 0 && i ~= j
                    if i + j < mid * 2
                        avg_x1 = avg_x1 + res(i, j) * i;
                        avg_y1 = avg_y1 + res(i, j) * j;
                        count1 = count1 + res(i, j);
                    else
                       avg_x2 = avg_x2 + res(i, j) * i;
                       avg_y2 = avg_y2 + res(i, j) * j;
                       count2 = count2 + res(i, j);
                    end
                end
            end
        end
    else
        for i = 1 : c
            for j = 1 : c
                if res(i, j) > 0 && i ~= j
                    if i > j
                        avg_x1 = avg_x1 + res(i, j) * i;
                        avg_y1 = avg_y1 + res(i, j) * j;
                        count1 = count1 + res(i, j);
                    else
                       avg_x2 = avg_x2 + res(i, j) * i;
                       avg_y2 = avg_y2 + res(i, j) * j;
                       count2 = count2 + res(i, j);
                    end
                end
            end
        end
    end
    avg_x1 = ceil(avg_x1 / count1);
    avg_y1 = ceil(avg_y1 / count1);
    avg_x2 = ceil(avg_x2 / count2);
    avg_y2 = ceil(avg_y2 / count2);
end


