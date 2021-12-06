function LeaderSequence = STS_LTS_Generator(type)
%UNTITLED3 此处显示有关此函数的摘要
%   此处显示详细说明
%  type参数: type=1：返回短符号+长符号；
%            type=2：返回短符号时域；
%            type=3：返回长符号时域；
%            type=4：返回短符号频域；
%            type=5：返回长符号频域；
%% [1]对理想信号处理
% STS频域表示，频点为-32~31，此处将52个频点外的零补全。
S=[0,0,0,0,0,0,0,0,1+1i,0,0,0,-1-1i,0,0,0,1+1i,0,0,0,-1-1i,0,...
    0,0,-1-1i,0,0,0,1+1i,0,0,0,0,0,0,0,-1-1i,0,0,0,-1-1i,0,0,...
    0,1+1i,0,0,0,1+1i,0,0,0,1+1i,0,0,0,1+1i,0,0,0,0,0,0,0];
% LTS频域表示，频点为-32~31，此处将52个频点外的零补全。
L=[0,0,0,0,0,0,1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,1,1,-1,-1,1,1,...
    -1,1,-1,1,1,1,1,0,1,-1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,...
    1,1,-1,-1,1,-1,1,-1,1,1,1,1,0,0,0,0,0];
% 保证OFDM符号的功率值稳定
S=sqrt(13/6)*S;
% 64点的傅里叶逆变换，变成时域
% 通过IFFT函数将STS频域的频点顺序调整为正频率（0~31）、负频率（-32~-1）
short=ifft(fftshift(S));
% 取前1~16个点，可以验证，后面17~32、33~48、49~64的数值都是1~16的复制
short_cp=short((1:16));
short=short_cp;
% 产生160个数据
for f=1:9
    short=[short,short_cp];
end
short_str = short;
% 长训练序列产生
% 通过IFFT函数将LTS频域的频点顺序调整为正频率（0~31）、负频率（-32~-1）
long_cp  = ifft(fftshift(L));
% 后32个数据
long1 = long_cp(33:64);
long2 = long_cp(1:64);
long_str = [long1,long2,long2];

preamble = [short_str,long_str];
% 对第一个和最后一个加窗处理，硬件上表现右移一位
% 第161个数据加窗处理
preamble(:,161) = preamble(:,161)*0.5 + preamble(:,1)*0.5;
% 第一个数据加窗处理
preamble(:,1) = preamble(:,1)*0.5;
% 短+长
if type == "ALL"
    LeaderSequence = preamble;
end
if type == "SHORT-T"
    LeaderSequence = short_str;
end
if type == "LONG-T"
    LeaderSequence = long_str;
end
if type == "SHORT-F"
    LeaderSequence = S;
end
if type == "LONG-F"
    LeaderSequence = L;
end
end