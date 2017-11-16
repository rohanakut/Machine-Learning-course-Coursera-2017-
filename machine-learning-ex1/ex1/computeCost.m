function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
%J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
X=X';
theta=theta';
y=y';
res=theta*X;
%j=res;
res=res-y;
res=res.^2;
res=sum(res); 
res=res/2;
res=res/m;
J=(res);
%J = (1/(2*m))*sum(power((X*theta - y),2));



% =========================================================================

end
