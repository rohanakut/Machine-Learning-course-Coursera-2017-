function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly
%clear;close all;clc;
%z=[0,59;50,32;-0.25,-777];
%g = zeros(size(z));
z=(-1).*z;
g=(1+exp(z));
g=g.^(-1);
%g=f;
%disp(g);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).





% =============================================================

end
