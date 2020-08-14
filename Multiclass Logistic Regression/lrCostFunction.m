function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 
m = length(y); 
h = sigmoid(X * theta);
unregularisedJ = -1/m * (y' * log(h) + (1 - y)' * log(1 - h));

a = theta;
a(2:length(a)) = a(2:length(a)) .^2;
a(1) = 0;
regExp = lambda/(2*m) * sum(a);
J = unregularisedJ + regExp;

regExp = (lambda/m)* theta(2 : length(theta));

dim = size(X);
temp1 = X(:, 2 : dim(2))';


multi = temp1 * (h - y);
grad = (1/m * multi) + regExp;

gradZero = 1/m * X(:, 1)' * (h - y);

grad = [gradZero; grad];

end
