%%
clear 
clc 
close all
%% generate data for 3 classes
x=[];
y=[];

figure(1)
i=1;
for mu=[-10 0 10]
mu1=[mu,mu];
sigma1 = [4 0; 0 3];
R = mvnrnd(mu1,sigma1,100);
plot(R(:,1),R(:,2),'+')
hold on

x=[x;R];
y0=zeros(100,3);
y0(:,i)=1;
y=[y;y0];
i=i+1;
end
axis([-20 20 -20 20])
x=[ones(size(x,1),1) x];

%% regression, middle set is mask by the other
b=pinv(x)*y;
% y_hat=x*b;
% % figure(2)
% % plot(y_hat,'-.');
figure(3)
fcontour(@(x,y)myfun(x,y,b),[-20,20],'-k','MeshDensity',400,'fill','on')
%% LDA
%mean
m1 = mean(x(1:100,2:3));
m2 = mean(x(101:200,2:3));
m3 = mean(x(201:300,2:3));
% s
s1= (x(1:100,2:3)-repmat(m1,100,1))'*(x(1:100,2:3)-repmat(m1,100,1));
s2= (x(101:200,2:3)-repmat(m2,100,1))'*(x(101:200,2:3)-repmat(m2,100,1));
s3= (x(201:300,2:3)-repmat(m3,100,1))'*(x(201:300,2:3)-repmat(m3,100,1));
s=(s1+s2+s3)/(300-3);
figure(4)
fcontour(@(x,y)myLDA(x,y,m1,m2,m3,s),[-20,20],'-k','MeshDensity',400,'fill','on')
%%
function yindex=myfun(x,y,b)
[~,yindex]=max([ones(size(x,1),1) x y]*b,[],2);
end

function yindex = myLDA(x,y,m1,m2,m3,s)
d1=[x,y]/s*m1'-0.5*m1/s*m1'+log(1/3);
d2=[x,y]/s*m2'-0.5*m2/s*m2'+log(1/3);
d3=[x,y]/s*m3'-0.5*m3/s*m3'+log(1/3);
[~,yindex]=max([d1 d2 d3],[],2);
end