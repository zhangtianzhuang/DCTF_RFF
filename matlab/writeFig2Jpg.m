function writeFig2Jpg(fig,filename)
    frame = getframe(fig);
    img = frame2im(frame);
    imwrite(img, filename);
    disp([filename, ', wirte success']);
end

