%Fetch data
load('./cloud.data')
labels = cloud(:,7);
cloud(:,7) = [];
trainX = cloud(1:800,:);
trainY = labels(1:800);
testX = cloud(801:end,:);
testY = labels(801:end);
lambdaVals = [0.01, 0.1, 0.3, 0.6, 1];

%##################
%LASSO Regression##
%##################

%Train using lasso for lambda values
B = lasso(trainX, trainY, 'lambda', lambdaVals);

%Print lambda values and number of non-zero coefficients
disp('Lambda Vals')
disp(lambdaVals)
numberOfNonZeros = sum(B~=0,1);
disp('Number of non-zeros')
disp(numberOfNonZeros)

%Calculate test error for lasso regression on each lambda val (Mean Squared
%Error)
estimatedVals = testX * B;
actualVals = repmat(testY, size(lambdaVals));
estimatedVals = estimatedVals - actualVals;
estimatedVals = estimatedVals.^2;
errorRates = mean(estimatedVals)

%Plot test error vs lambda values
plot(lambdaVals, errorRates), xlabel('Lambda Values'), ylabel('Error Rate'), title('Lasso Regression Data B')

%##################
%Ridge Regression##
%##################

%Run ridge regression on lambda values
lambdaVals = [1, 20, 40, 60, 80];
B = ridge(trainY, trainX, lambdaVals);

%Print lambda values and number of non-zero coefficients
disp('Lambda Vals')
disp(lambdaVals)
numberOfNonZeros = sum(B~=0,1);
disp('Number of non-zeros')
disp(numberOfNonZeros)

%Calculate test error for ridge regression on each lambda val (Mean Squared
%Error)
estimatedVals = testX * B;
actualVals = repmat(testY, size(lambdaVals));
estimatedVals = estimatedVals - actualVals;
estimatedVals = estimatedVals.^2;
errorRates = mean(estimatedVals)

%Plot test error vs lambda values
figure;
plot(lambdaVals, errorRates), xlabel('Lambda Values'), ylabel('Error Rate'), title('Ridge Regression Data B')


