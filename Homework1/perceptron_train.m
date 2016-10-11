function [theta,k] = perceptron_train( X, y )
%PERCEPTRON_TRAIN trains a perceptron classifier on a training set of n
%examples, each of which is a d dimensional vector. The labels of the
%examples are in y and are -1 or 1. The function returns theta, k, the
%final classification vector and the number of updates permormed,
%respectively. This is the simple perceptron classifier seen in class,
%where the linear separator passes through the origin. X and y are n x d
%and n x 1 matrices respectively.
%   Detailed explanation goes here

%instantiate theta
theta = y(1) .* X(1,:);
%count number of updates
k = 0;
converge = 0;
%iterate over dataset multiple times until convergence (won't work with all
%datasets, but will work for homework)
while converge == 0
    converge = 1;
    for i=1:size(y,1)
        %check for error
        m = dot(X(i,:),theta);
        if (y(i) * m < 0)
            converge = 0;
            %disp(y(i) * m);
            k = k+1;
            theta = theta + y(i) .* X(i,:);
        end
    end
end
end

