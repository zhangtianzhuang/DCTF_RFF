function img = graycolor(plotTable)
min_n=min(min(plotTable));
max_n=max(max(plotTable));
plotTable=plotTable./(max_n-min_n)*255;
[M,N]=size(plotTable);
for i=1:1:M
    for j=1:1:N
%         if plotTable(i, j) < 100
%             plotTable(i, j) = 0;
%         else
%             plotTable(i, j) = 230;
%         end
        R(i,j)=GrayColorR(plotTable(i,j));
        G(i,j)=GrayColorG(plotTable(i,j));
        B(i,j)=GrayColorB(plotTable(i,j)); 
    end  
end  
img(1:1:M,1:1:N,1)=R(1:M,1:N);  
img(1:1:M,1:1:N,2)=G(1:M,1:N);  
img(1:1:M,1:1:N,3)=B(1:M,1:N);  


function r=GrayColorR(gray)  
r=0;  
if gray>=170  
    r=255;
end  
if gray>=128&&gray<=170  
    r=255/42*(gray-128);  
end  
return;  

function g=GrayColorG(gray)  
g=0;  
if gray >=84 && gray<=170  
    g=255;
end  
if gray<=84  
    g=255/84*gray;  
end  
if gray>=170&&gray<=255  
    g=255/85*(255-gray);  
end  
return;  

function b=GrayColorB(gray)
b=0;  
if gray<=84  
    b=255;  
end  
if gray>=84&&gray<=128  
    b=255/44*(128-gray);
end  
return;  
