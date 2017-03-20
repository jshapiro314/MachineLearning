%Fetch data
trainX = randn(1000, 20);
trainY = randn(1000, 1);
testX = randn(500, 20);
testY = randn(500, 1);
lambdaVals = [0.01, 0.05, 0.1, 0.2, 0.3];

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
plot(lambdaVals, errorRates), xlabel('Lambda Values'), ylabel('Error Rate'), title('Lasso Regression Data A')

%##################
%Ridge Regression##
%##################

%Run ridge regression on lambda values (which are changed for dataset A)
lambdaVals = [1, 50, 100, 200, 1000];
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
plot(lambdaVals, errorRates), xlabel('Lambda Values'), ylabel('Error Rate'), title('Ridge Regression Data A')


