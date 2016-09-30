close all;
clear;
clc;

load('mnist_49_3000.mat');
load('result.mat');
x = x(:,2001:3000);
y = y(:,2001:3000);
[d,n] = size(x);
k = [1:n];
errorList = k(theta_x.*y<0);

index = [1:size(errorList,2)];
chosen_theta = theta_x(errorList);
[A,I] = sort(abs(chosen_theta), 'descend');
imgList = errorList(index(I));

set(gca,'FontSize', 10)
count = 0;
for i=1:4
    for j=1:5
        count = count+1;
        subplot(4,5,(i-1)*5+j);
        conf = theta_x(imgList(count));
        name = ['label: ', num2str(y(imgList(count))), ' | confidence: ', num2str(conf)];
        imagesc(reshape(x(:,imgList(count)),[sqrt(d),sqrt(d)]));
        title(name);
    end
end