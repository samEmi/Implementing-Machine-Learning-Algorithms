function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z, which can be a matrix,
%               vector or scalar.

denom = 1 + exp(-z);


g = 1 ./ denom;



end
