function [x_us] = randu_sphere(N,M,maxV,minV)
%This function generate Uniformly Sampled Points on the Unitary
%N-Dimensional Sphere


x = (maxV-minV)*rand(N,M)+minV; % M vectors of N independent standard normal variates
x_us = x./repmat(sqrt(sum(x.^2,1)),N,1); % Project onto the N-1 dimensional surface of the N-dimensional unit sphere

%
% figure(),
% set(gcf,'Color','white')
% plot(x(1,:),x(2,:),'.r'), hold on
% plot(x_us(1,:),x_us(2,:),'.k'), hold on
% box off
% axis equal

end