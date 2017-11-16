function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m=length(y);
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
   % x=length(y);
   % J_history=zeros(num_iters,1);
    %for i=1:num_iters
        
            a=X*theta-y;
            delta=1/m*(a'*X);
            %A=sum(a);
            delta1=(alpha)*delta;
            theta=theta-delta1';
            %theta(2)=theta(2)-A;

            

    % ============================================================

    % Save the cost J in every iteration    
   J_history(iter) = computeCost(X, y, theta);

end

end
