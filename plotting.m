close all;
clear;
clc;

load('trainingDetail.mat');

n = size(gradient_List,2);
x =[1:1:n];
plot(db(x), db(error_List), db(x), db(gradient_List), db(x), db(gradient_diff_List));
legend('x','y','z')