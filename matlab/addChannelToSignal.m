function signal = addChannelToSignal(signal, channel)
    [~, n] = size(signal);
    for i = 1:n
        x = signal(:, i);
        y = channel(x);
        signal(:, i) = y;
    end
end
