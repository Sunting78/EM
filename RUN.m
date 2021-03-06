% clear all;
% I=imread('S2001L01.jpg');
% % I=imread('1.jpg');
% imshow(I)
% [h,w]=size(I);
% imhist(I);
% i=I(:);
% W1=[0.1,0.2,0.4,0.3];
% M1=[20,80,120,160];
% V1(1,1,1)=15;
% V1(1,1,2)=25;
% V1(1,1,3)=20;
% V1(1,1,4)=20;
% s.W=W1;
% s.M=M1;
% s.V=V1;
% [W,M,V,L] = EM_GM_fast(double(i),4,[],100,1,s);
% 
% for i=1:h
%     for j=1:w
%         y=[];
%         for k=1:4
%             y(k)=W(1,k)*normpdf(double(I(i,j)),M(1,k),sqrt(V(1,1,k)));%为什么都没有第二类？
%             %         y(k)=normpdf(double(I(i,j)),M(1,k),V(1,1,k));
%         end
%         [~,x]=find(y==max(y));
%         
%         I(i,j)=x;
%     end
% end
% imshow(I,[]);

clear all;

I=imread('1.jpg');
imshow(I)
[h,w]=size(I);
imhist(I);
i=I(:);
W1=[0.08,0.18,0.24,0.5];
M1=[20,80,110,140];
V1(1,1,1)=15;
V1(1,1,2)=10;
V1(1,1,3)=10;
V1(1,1,4)=20;
s.weight=W1;
s.mean=M1;
s.var=V1;

[W,M,V] = EM_GMM(double(i),4,s,100);

for i=1:h
    for j=1:w
        y=[];
        for k=1:4
            y(k)=W(1,k)*normpdf(double(I(i,j)),M(1,k),sqrt(V(1,1,k)));%为什么都没有第二类？
            %         y(k)=normpdf(double(I(i,j)),M(1,k),V(1,1,k));
        end
        [~,x]=find(y==max(y));
        
        I(i,j)=x;
    end
end
imshow(I,[]);