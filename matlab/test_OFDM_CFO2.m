clear 
close all 
clc 

clc
clear
NbSymb = 6;
Nfft=64; %FFT的大小
Ncp=16;
CFO=0.07;%归一化频偏大小
% h= modem.pskmod('M', 2); 
% h2= modem.pskdemod('M', 2); 
M = 2;
nu = log2(M);
msg=randi([0 M-1],1,Nfft*NbSymb);
modmsg = pskmod(msg,M);
% modmsg=modulate(h,msg); %BPSK调制
% modmsg = qammod(msg,M);
% nf = modnorm(modmsg,'avpow',1);
% modmsg = nf*modmsg;
figure;
plot(modmsg,'rx','MarkerSize',8,'LineWidth',2)
axis([-1.2 1.2 -1 1])
grid on
title('transmitted signal')
xlabel('In-phase')
ylabel('Quadrature')
modmsg = reshape(modmsg, Nfft, NbSymb);
modmsg1 = modmsg(:,1);
for k=2:1:NbSymb
    modmsg(:,k) = modmsg1;
end
x_ifft=sqrt(Nfft)*ifft(modmsg,Nfft); %频域变时域
x_CP = add_CP(x_ifft,Nfft,Ncp);
x = x_CP(:);
temp = [1+0.8i 0.5-0.3i 0.1+0.1i];
h = temp/sqrt(sum(abs(temp).^2));
% h = 1;
r = conv(x,h);
% snr = 30
% r = awgn(r(1:(Nfft+Ncp)*NbSymb),snr,'measured');
r = r(1:(Nfft+Ncp)*NbSymb);

y = add_CFO(r,CFO,Nfft);%加入频偏
y = reshape(y, Nfft+Ncp, NbSymb);

y_noCP=remove_CP(y,Nfft,Ncp);
y_fft=fft(y_noCP)/sqrt(Nfft);

aaa = calculateCFO(y);

disp(aaa);

CFO_est = CFO;
alp=sin(pi*CFO_est)*exp(1i*pi*CFO_est*(Nfft-1)/Nfft)/(Nfft*sin(pi*CFO_est/Nfft));  %幅度以及相位的影响
a=exp(1i*pi*CFO_est*(Nfft-1)/Nfft);
for k=1:Nfft
    sum=0;
    for m=1:Nfft
        if m==k
            sum=sum+0;
        else
            sum=sum+(sin(pi*(m-k+CFO_est))/(Nfft*sin(pi*(m-k+CFO_est)/Nfft)))*modmsg(m)*exp(1i*pi*(m-k)*(Nfft-1)/Nfft);
        end
    end
    I(k)=a.*sum; %计算I（k），子载波之间干扰
end

figure
% plot(y_fft(:,1),'x','MarkerSize',10,'LineWidth',2)
hold on 
plot(y_fft(:,1:end),'LineStyle', 'none','Marker', '+')
axis([-1.5 1.5 -1.5 1.5])
grid on
y_diff = y_fft(:,1:end-1).*conj(y_fft(:,2:end));
plot(y_diff,'LineStyle', 'none','Marker', '.')

% y_diff12 = y_fft2.*conj(y_fft1);

%% 理论推导得到的相邻OFDM符号同一子载波上的频偏
phase = exp(-1i*2*pi*CFO*(Nfft+Ncp)/Nfft);

plot(phase,'r+','MarkerSize',8,'LineWidth',2)
% grid on
legend('OFDM Symbol 1','OFDM Symbol 2','OFDM Symbol 3','OFDM Symbol 4', ...
    'OFDM Symbol 5','OFDM Symbol 6','differential','theoretical phase rotation','Location','southeast')
title('differential w/o compensation')
xlabel('In-phase')
ylabel('Quadrature')
box on

%% compensation |H|
H = y_fft(:,1)./modmsg(:,1);
figure 
hold on
for k = 1:1:NbSymb
    Hest(:,k) = y_fft(:,k)./modmsg(:,1);
    plot(real(Hest(:,k) ))

end
axis([0 Nfft 0 1.5])
title('channel frequency response (real part)');
xlabel('subcarrier')
ylabel('amplitude')
grid on
box on


figure; hold on
for k = 1:1:NbSymb
    plot(imag(Hest(:,k) ))
end

axis([0 Nfft 0 1.5])
title('channel frequency response (imaginary part)');
xlabel('subcarrier')
ylabel('amplitude')
box on
grid on

for k = 1:1:NbSymb
    y_fft_comp(:,k) = y_fft(:,k)./H;
    y_fft_comp2(:,k) = y_fft(:,k)./abs(H);
end
y_diff_comp = y_fft_comp(:,1:end-1).*conj(y_fft_comp(:,2:end));
y_diff_comp2 = y_fft_comp2(:,1:end-1).*conj(y_fft_comp2(:,2:end));

figure
% plot(y_fft(:,1),'x','MarkerSize',10,'LineWidth',2)
hold on 
plot(y_diff_comp,'LineStyle', 'none','Marker', '.')
axis([-1.5 1.5 -1.5 1.5])
plot(phase,'r+','MarkerSize',8,'LineWidth',2)
grid on
legend('differential','Location','southeast')
title('differential w/o compensation')
xlabel('In-phase')
ylabel('Quadrature')
box on

figure
hold on 
plot(y_diff_comp2,'LineStyle', 'none','Marker', '.')
axis([-1.5 1.5 -1.5 1.5])
plot(phase,'r+','MarkerSize',8,'LineWidth',2)
grid on
legend('differential','Location','southeast')
title('differential w/ |H| compensation')
xlabel('In-phase')
ylabel('Quadrature')
box on

%% 
Plot_Table_Size = 64;  % DCTF Size
Plot_Table_Max = 1.4;  % DCTF Size 
Process_Data_Offset = 0; % Delete some samples for generating DCTF
d = y_diff_comp/sqrt(mean(mean(abs(y_diff_comp.^2))));
Data_Process_Raw_Offset = d(:);
Plot_Table = F_Get_Data_Table(Plot_Table_Size,Plot_Table_Max,Data_Process_Raw_Offset,Process_Data_Offset);
min_table=min(min(Plot_Table));  % minmum of the matrix
max_table=max(max(Plot_Table));  % maxmum of the matrix
Plot_Table=Plot_Table./(max_table-min_table)*255;

figure;
graycolor(Plot_Table,1,'f_0_05_comp.jpg');

%% 
Plot_Table_Size = 64;  % DCTF Size
Plot_Table_Max = 1.4;  % DCTF Size 
Process_Data_Offset = 0; % Delete some samples for generating DCTF
d = y_diff_comp2/sqrt(mean(mean(abs(y_diff_comp2.^2))));
Data_Process_Raw_Offset = d(:);
Plot_Table = F_Get_Data_Table(Plot_Table_Size,Plot_Table_Max,Data_Process_Raw_Offset,Process_Data_Offset);
min_table=min(min(Plot_Table));  % minmum of the matrix
max_table=max(max(Plot_Table));  % maxmum of the matrix
Plot_Table=Plot_Table./(max_table-min_table)*255;

figure;
graycolor(Plot_Table,1,'f_0_05.jpg');



figure
plot(real(modmsg(:,1)))
hold on 
plot(real(y_fft_comp(:,1)));
plot(real(y_fft_comp2(:,1)));
legend('original','compensation H', 'compensation |H|')
title('real part')
figure
plot(imag(modmsg(:,1)))
hold on 
plot(imag(y_fft_comp(:,1)));
plot(imag(y_fft_comp2(:,1)));
legend('original','compensation H', 'compensation |H|')
title('imaginary part')


figure
plot(real(modmsg(:,2)))
hold on 
plot(real(y_fft_comp(:,2)));
plot(real(y_fft_comp2(:,2)));
legend('original','compensation H', 'compensation |H|')
title('real part')
figure
plot(imag(modmsg(:,2)))
hold on 
plot(imag(y_fft_comp(:,2)));
plot(imag(y_fft_comp2(:,2)));
legend('original','compensation H', 'compensation |H|')
title('imaginary part')


figure
plot(real(modmsg(:,3)))
hold on 
plot(real(y_fft_comp(:,3)));
plot(real(y_fft_comp2(:,3)));
legend('original','compensation H', 'compensation |H|')
title('real part')
figure
plot(imag(modmsg(:,3)))
hold on 
plot(imag(y_fft_comp(:,3)));
plot(imag(y_fft_comp2(:,3)));
legend('original','compensation H', 'compensation |H|')
title('imaginary part')


figure
plot(real(modmsg(:,4)))
hold on 
plot(real(y_fft_comp(:,4)));
plot(real(y_fft_comp2(:,4)));
legend('original','compensation H', 'compensation |H|')
title('real part')
figure
plot(imag(modmsg(:,4)))
hold on 
plot(imag(y_fft_comp(:,4)));
plot(imag(y_fft_comp2(:,4)));
legend('original','compensation H', 'compensation |H|')
title('imaginary part')


theta = angle(y_diff12);
figure; plot(theta)

x_compe=((y_fft-I)./alp)'  %补偿之后的信号
xdemod=pskdemod(x_compe,M);
figure
plot(x_compe,'.')

%实现对OFDM加入频偏
function y_CFO=add_CFO(y,CFO,Nfft)
nn=(0:length(y)-1).';
y_CFO=y.*exp(1i*2*pi*CFO*nn/Nfft);
end

%实现对OFDM加入CP
function y_CP=add_CP(y,Nfft,Ncp)
% y_CP=[y(end-Ncp+1:end) y];
y_CP=[y(end-Ncp+1:end, :); y];
end

%实现对OFDM去掉CP
function y=remove_CP(y_CP,Nfft,Ncp)
% y=y_CP(Ncp+1:end, :);
y=y_CP(Ncp+1:end, :);

end

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