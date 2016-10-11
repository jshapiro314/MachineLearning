%load necessary data
load('MNISTdata.mat')

%calculate means based on training data and labels
M = calculateClassMean(trainx,trainLabel);
%initialize labels for M
MLabels = zeros(10,1);
for i = 1:10
    MLabels(i) = (i-1);
end

%initialize vector of k for plot
k = [1,3,5,7,9,11,13,15,17,19];
%initialize vector for errors to plot against k
y = zeros(1,10);
meanError = 0;
%Iterate over test data using knn with training data and just means
for i=1:testLabel
    for j=1:10
        [class,ids] = knn(trainx,trainLabel,testx(i,:),k(j));
        if class ~= testLabel(i)
            y(j) = y(j)+1;
        end
    end
    
    [class,ids] = knn(M,MLabels,testx(i,:),1);
    if class ~= testLabel(i)
        meanError = meanError+1;
    end
end

%calculated based on 1000 element test set
meanError = meanError / 1000;
y = y ./ 1000;

fprintf('Error for K values:\n');
disp(k);
disp(y);
fprintf('Error for using means instead of training data:\n');
disp(meanError);
plot(k,y);
%plot(1,meanError);
