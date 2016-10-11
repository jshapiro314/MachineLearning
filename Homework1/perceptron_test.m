function [ testErr ] = perceptron_test( theta, XTest, yTest )
%PERCEPTRON_TEST returns the fraction of test examples which were
%misclassified. Theta is the classification vector to be used, XTest and
%yTest are m x d and m x 1 matrices respectively, corresponding to m test
%examples and their true labels.
%   Detailed explanation goes here

%count number of wrong classifications
testErr = 0;

for i = 1:size(yTest,1)
   %check for error
   m = dot(XTest(i,:),theta);
   if (yTest(i) * m < 0)
       testErr = testErr + 1;
   end
end

testErr = testErr / size(yTest,1);


end

