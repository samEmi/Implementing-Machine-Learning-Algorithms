function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class.

m = size(X, 1);
p = zeros(size(X, 1), 1);

X = [ones(m, 1) X];
A = sigmoid(X * all_theta');


for i = 1:m
    maximum = max(A(i, :));
    prediction = find(A(i, :) == maximum);
    p(i) = prediction;
end









% =========================================================================


end
