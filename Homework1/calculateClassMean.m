function [ M ] = calculateClassMean( X, labels )
%CALCULATECLASSMEAN Takes in matrix X which includes all elements in
%training set as well as label which inclues each element's classification.
%Outputs 10x784 matrix of mean value of elements in the specified
%class. the ith row will be the mean of the i-1 label.
%   Detailed explanation goes here
totals = zeros(10,784);
counts = zeros(10,1);

%get counts and totals
for i=1:size(labels,1)
   counts(labels(i)+1) = counts(labels(i)+1)+1;
   totals(labels(i)+1,:) = totals(labels(i)+1,:) + X(i,:);
end

%make some averages
for i=1:784
    %disp(totals(:,i))
    totals(:,i) = totals(:,i) ./ counts;
    %disp(totals(:,i))
end

M=totals;

