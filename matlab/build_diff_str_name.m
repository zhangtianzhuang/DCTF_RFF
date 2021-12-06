function res =  build_diff_str_name(diff)
    res = num2str(diff(1));
    for i = 2 : length(diff)
        res = [res, '_', num2str(diff(i))];
    end
end