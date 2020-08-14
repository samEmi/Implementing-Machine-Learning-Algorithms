function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)
m = size(X, 1);
num_labels = size(Theta2, 1);
p = zeros(size(X, 1), 1);
X = [ones(m, 1) X];

hidden = Theta1 * X';
hidden = sigmoid(hidden);

hidden = [ones(m, 1) hidden'];

result = Theta2 * hidden';
A = sigmoid(result)';

for i = 1:m
    maximum = max(A(i, :));
    prediction = find(A(i, :) == maximum);
    p(i) = prediction;
end

end
