function [ output ] = kernelPerceptron( train_data, train_labels, test_data, test_labels, kernel )
%KERNELPERCEPTRON ASSUMES points on classifier are negative. During
%training, we run over all data 20 times.
%   outputs array of labels for test data, -1 & 1 are only values
%   train_data: each column is a feature, each row is datapoint
%   train_labels: vector of labels corresponding to train_data, same length
%   test_data: Each column is a feature, each row is datapoint
%   test_labels: vector of labels corresponding to test_data, same length
%   kernel: Function handle to input which kernel to use. Takes in 2
%   vectors

%TRAIN

%initialize alpha vector of size of training data
alpha = zeros(size(train_labels));
for j=1:20
for t=1:size(train_labels)
    %calculate current theta
    theta = 0;
    for i=1:size(train_labels)
        theta = alpha(i) * train_labels(i) * kernel(train_data(i),train_data(t)) + theta;
    end
    
    %check if perceptron made a mistake
    theta = theta * train_labels(t);
    if (theta <= 0)
        alpha(t) = alpha(t) + 1;
    end
end
end
%TEST

%initialize vector to hold results
output = zeros(size(test_labels));

for i=1:size(test_labels)
    %calculate output
    outputVal = 0;
    
    for j=1:size(train_labels)
        outputVal = alpha(j) * train_labels(j) * kernel(train_data(j),test_data(i)) + outputVal;
    end
    
    %check for output
    if (outputVal <= 0)
        output(i) = -1;
    else
        output(i) = 1;
    end
end


end

