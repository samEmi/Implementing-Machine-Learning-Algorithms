function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); 

probabilities = X * theta;
for i = 1 : m
    if probabilities(i) >= 0
        probabilities(i) = 1;
    else
        probabilities(i) = 0;
    end
end

p = probabilities;


end
