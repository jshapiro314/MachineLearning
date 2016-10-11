load_p1_a

[theta,k] = perceptron_train(X,y);
fprintf('The number of updates K on the training data: \n');
disp(k);
fprintf('Theta: \n');
disp(theta);
angle = atan2d(theta(2), theta(1)) - atan2d(0, 1)
fprintf('TestErr on training set: \n');
testErr = perceptron_test(theta,X,y);
disp(testErr);
%calculate margins
margins = zeros(1,size(y,1));
%i = 1:1:size(y,1);
for i=1:size(y,1)
    %dot(X(i,:),theta)
    %norm(X(i,:))
    margins(i) = abs((dot(X(i,:),theta))/(norm(X(i,:))));
end
minMargin = min(margins)
load_p1_b
%preprocess training data
X(:,3) = 1;
[theta,k] = perceptron_train(X,y);
fprintf('The number of updates K on the training data for IRIS dataset: \n');
disp(k);
fprintf('Theta: \n');
disp(theta);
A = [1,0,0];
A = A';
angle = atan2d(norm(cross(A,theta)),dot(A,theta))
fprintf('TestErr on training set: \n');
%testErr = perceptron_test(theta,X,y);
disp(testErr);
for i=1:size(y,1)
    %dot(X(i,:),theta)
    %norm(X(i,:))
    margins(i) = abs((dot(X(i,:),theta))/(norm(X(i,:))));
end
minMargin = min(margins)