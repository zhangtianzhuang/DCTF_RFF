function save_png(fig, filename)
    % 以.png格式保存图片到文件中，默认高清600dpi
    % fig保存要绘制的fig
    % filename 图片全路径
    print(fig, '-dpng', '-r600', filename);
    disp(['the figure has been stored at the disk [', filename,  '] successfully']);
end