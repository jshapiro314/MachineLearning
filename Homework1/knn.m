function [ class, ids ] = knn( X,C,z,k )
%KNN X is the training data, C is a vector of labels, values returned are
%indecies of the k nearest neighbors (ids) and the class the object falls in
%(by majority vote). NOTE: I am assuming z is a 784 dimmension vector, not
%an index to a vector in the testSet.
%   Detailed explanation goes here

sqlength = sum(X .* X, 2);
distArray = sqlength - 2 * X * z';
ids = zeros(1,k);
votes = zeros(1,10);

maxDist = max(distArray);
for i=1:k
    [x,I] = min(distArray);
    %make sure we don't find the same min again...
    distArray(I) = abs(x) + maxDist;
    ids(i) = I;
    votes(C(I)+1) = votes(C(I)+1) + 1;
end

%fetch majority vote
[x,class] = max(votes);
class = class - 1;

end

