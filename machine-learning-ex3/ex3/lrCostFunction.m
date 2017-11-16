function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%
theta_x=zeros(length(y),1);
    theta_x=X*theta;
    hypo_func=sigmoid(theta_x);
    A=log(hypo_func);
    B=log(1-hypo_func);
     y=y';
    j_theta=((y*A)+(1-y)*(B));
     j_theta=(-1/m)*j_theta;
    y=y';
    grad=(hypo_func-y)';
    grad=(1/m)*(grad*X);
    %disp(grad);
    theta1=theta;
    theta1=theta1.^2;
    theta1(1,1)=0;
    theta(1,1)=0;
    %disp(theta1);
    regu=sum(theta1);
    %disp(regu);
    regu=(lambda/(2*m))*regu;
    %disp(regu);
    j_theta=j_theta+regu;
    grad_regu=(lambda/m)*(theta);
    grad_regu=(grad_regu)';
    grad=grad+(grad_regu);
    J=j_theta;
    %disp(grad);
    %disp(j_theta);

% =============================================================

grad = grad(:);

end
