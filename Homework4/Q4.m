%Load in data
load('./USPS/0vAllData.mat');
load('./USPS/0vAllLabels.mat');

%Create validation set (first 1/6 of data)
valLen = int64(size(newlabels,1)/6);
validationData = newdata(1:valLen,:);
validationLabels = newlabels(1:valLen);

%Split validation set into train and test (2/3)
valTrainLen = int64(valLen * 2 / 3);
valTrainData = validationData(1:valTrainLen,:);
valTrainLabels = validationLabels(1:valTrainLen);
valTestData = validationData(valTrainLen+1:end,:);
valTestLabels = validationLabels(valTrainLen+1:end);

%create SVM classifiers for linear kernel
lin001Model = svmtrain(valTrainData, valTrainLabels, 'autoscale', 'false', 'boxconstraint', 0.01);
lin01Model = svmtrain(valTrainData, valTrainLabels, 'autoscale', 'false', 'boxconstraint', 0.1);
lin1Model = svmtrain(valTrainData, valTrainLabels, 'autoscale', 'false', 'boxconstraint', 1);
lin20Model = svmtrain(valTrainData, valTrainLabels, 'autoscale', 'false', 'boxconstraint', 20);
lin50Model = svmtrain(valTrainData, valTrainLabels, 'autoscale', 'false', 'boxconstraint', 50);

%create SVM classifiers for polynomial kernel
poly001Model = svmtrain(valTrainData, valTrainLabels, 'kernel_function', 'polynomial', 'autoscale', 'false', 'boxconstraint', 0.01);
poly01Model = svmtrain(valTrainData, valTrainLabels, 'kernel_function', 'polynomial', 'autoscale', 'false', 'boxconstraint', 0.1);
poly1Model = svmtrain(valTrainData, valTrainLabels, 'kernel_function', 'polynomial', 'autoscale', 'false', 'boxconstraint', 1);
poly20Model = svmtrain(valTrainData, valTrainLabels, 'kernel_function', 'polynomial', 'autoscale', 'false', 'boxconstraint', 20);
poly50Model = svmtrain(valTrainData, valTrainLabels, 'kernel_function', 'polynomial', 'autoscale', 'false', 'boxconstraint', 50);

%create SVM classifiers for gaussian rbf kernel
rbf001Model = svmtrain(valTrainData, valTrainLabels, 'kernel_function', 'rbf', 'autoscale', 'false', 'boxconstraint', 0.01);
rbf01Model = svmtrain(valTrainData, valTrainLabels, 'kernel_function', 'rbf', 'autoscale', 'false', 'boxconstraint', 0.1);
rbf1Model = svmtrain(valTrainData, valTrainLabels, 'kernel_function', 'rbf', 'autoscale', 'false', 'boxconstraint', 1);
rbf20Model = svmtrain(valTrainData, valTrainLabels, 'kernel_function', 'rbf', 'autoscale', 'false', 'boxconstraint', 20);
rbf50Model = svmtrain(valTrainData, valTrainLabels, 'kernel_function', 'rbf', 'autoscale', 'false', 'boxconstraint', 50);

%classify test data with linear kernel
lin001Results = svmclassify(lin001Model, valTestData);
lin01Results = svmclassify(lin01Model, valTestData);
lin1Results = svmclassify(lin1Model, valTestData);
lin20Results = svmclassify(lin20Model, valTestData);
lin50Results = svmclassify(lin50Model, valTestData);

%classify test data with polynomial kernel
poly001Results = svmclassify(poly001Model, valTestData);
poly01Results = svmclassify(poly01Model, valTestData);
poly1Results = svmclassify(poly1Model, valTestData);
poly20Results = svmclassify(poly20Model, valTestData);
poly50Results = svmclassify(poly50Model, valTestData);

%classify test data with gaussian rbf kernel
rbf001Results = svmclassify(lin001Model, valTestData);
rbf01Results = svmclassify(lin01Model, valTestData);
rbf1Results = svmclassify(lin1Model, valTestData);
rbf20Results = svmclassify(lin20Model, valTestData);
rbf50Results = svmclassify(lin50Model, valTestData);

%print out error rate
%if we add our guesses with the actual labels, the incorrect ones will = 0
%and the correct ones will either equal -2 or 2. So the percentage of 0s =
%error rate.

%Linear
guessedLabels = lin001Results + valTestLabels;
errorRateLin001 = 1 - (nnz(guessedLabels) ./ size(valTestLabels,1))

guessedLabels = lin01Results + valTestLabels;
errorRateLin01 = 1 - (nnz(guessedLabels) ./ size(valTestLabels,1))

guessedLabels = lin1Results + valTestLabels;
errorRateLin1 = 1 - (nnz(guessedLabels) ./ size(valTestLabels,1))

guessedLabels = lin20Results + valTestLabels;
errorRateLin20 = 1 - (nnz(guessedLabels) ./ size(valTestLabels,1))

guessedLabels = lin50Results + valTestLabels;
errorRateLin50 = 1 - (nnz(guessedLabels) ./ size(valTestLabels,1))

%Polynomial
guessedLabels = poly001Results + valTestLabels;
errorRatePoly001 = 1 - (nnz(guessedLabels) ./ size(valTestLabels,1))

guessedLabels = poly01Results + valTestLabels;
errorRatePoly01 = 1 - (nnz(guessedLabels) ./ size(valTestLabels,1))

guessedLabels = poly1Results + valTestLabels;
errorRatePoly1 = 1 - (nnz(guessedLabels) ./ size(valTestLabels,1))

guessedLabels = poly20Results + valTestLabels;
errorRatePoly20 = 1 - (nnz(guessedLabels) ./ size(valTestLabels,1))

guessedLabels = poly50Results + valTestLabels;
errorRatePoly50 = 1 - (nnz(guessedLabels) ./ size(valTestLabels,1))

%Gaussian rbf
guessedLabels = rbf001Results + valTestLabels;
errorRaterbf001 = 1 - (nnz(guessedLabels) ./ size(valTestLabels,1))

guessedLabels = rbf01Results + valTestLabels;
errorRaterbf01 = 1 - (nnz(guessedLabels) ./ size(valTestLabels,1))

guessedLabels = rbf1Results + valTestLabels;
errorRaterbf1 = 1 - (nnz(guessedLabels) ./ size(valTestLabels,1))

guessedLabels = rbf20Results + valTestLabels;
errorRaterbf20 = 1 - (nnz(guessedLabels) ./ size(valTestLabels,1))

guessedLabels = rbf50Results + valTestLabels;
errorRaterbf50 = 1 - (nnz(guessedLabels) ./ size(valTestLabels,1))

%Based on data above, run cross validation using linear 50, polynomial 50,
%rbf 50


%Partition remaining 5/6 of data into quarters for cross validation
crossValLen = int64((size(newlabels,1)-valLen)/4);

crossValData1 = newdata(valLen+1:valLen+crossValLen,:);
crossValLabels1 = newlabels(valLen+1:valLen+crossValLen);

crossValData2 = newdata(valLen+crossValLen+1:valLen+crossValLen*2,:);
crossValLabels2 = newlabels(valLen+crossValLen+1:valLen+crossValLen*2);

crossValData3 = newdata(valLen+crossValLen*2+1:valLen+crossValLen*3,:);
crossValLabels3 = newlabels(valLen+crossValLen*2+1:valLen+crossValLen*3);

crossValData4 = newdata(valLen+crossValLen*3+1:end,:);
crossValLabels4 = newlabels(valLen+crossValLen*3+1:end);

%Create different training sets
trainingData1 = [crossValData2; crossValData3; crossValData4];
trainingLabels1 = [crossValLabels2; crossValLabels3; crossValLabels4];

trainingData2 = [crossValData1; crossValData3; crossValData4];
trainingLabels2 = [crossValLabels1; crossValLabels3; crossValLabels4];

trainingData3 = [crossValData1; crossValData2; crossValData4];
trainingLabels3 = [crossValLabels1; crossValLabels2; crossValLabels4];

trainingData4 = [crossValData1; crossValData2; crossValData3];
trainingLabels4 = [crossValLabels1; crossValLabels2; crossValLabels3];

%train linear classifiers
lin50Model1 = svmtrain(trainingData1, trainingLabels1, 'autoscale', 'false', 'boxconstraint', 50);
lin50Model2 = svmtrain(trainingData2, trainingLabels2, 'autoscale', 'false', 'boxconstraint', 50);
lin50Model3 = svmtrain(trainingData3, trainingLabels3, 'autoscale', 'false', 'boxconstraint', 50);
lin50Model4 = svmtrain(trainingData4, trainingLabels4, 'autoscale', 'false', 'boxconstraint', 50);

%train polynomial classifiers
poly50Model1 = svmtrain(trainingData1, trainingLabels1, 'kernel_function', 'polynomial', 'autoscale', 'false', 'boxconstraint', 50);
poly50Model2 = svmtrain(trainingData2, trainingLabels2, 'kernel_function', 'polynomial', 'autoscale', 'false', 'boxconstraint', 50);
poly50Model3 = svmtrain(trainingData3, trainingLabels3, 'kernel_function', 'polynomial', 'autoscale', 'false', 'boxconstraint', 50);
poly50Model4 = svmtrain(trainingData4, trainingLabels4, 'kernel_function', 'polynomial', 'autoscale', 'false', 'boxconstraint', 50);

%train rbf classifiers
rbf50Model1 = svmtrain(trainingData1, trainingLabels1, 'kernel_function', 'rbf', 'autoscale', 'false', 'boxconstraint', 50);
rbf50Model2 = svmtrain(trainingData2, trainingLabels2, 'kernel_function', 'rbf', 'autoscale', 'false', 'boxconstraint', 50);
rbf50Model3 = svmtrain(trainingData3, trainingLabels3, 'kernel_function', 'rbf', 'autoscale', 'false', 'boxconstraint', 50);
rbf50Model4 = svmtrain(trainingData4, trainingLabels4, 'kernel_function', 'rbf', 'autoscale', 'false', 'boxconstraint', 50);

%Test linear classifiers
lin50Results1 = svmclassify(lin50Model1, crossValData1);
lin50Results2 = svmclassify(lin50Model2, crossValData2);
lin50Results3 = svmclassify(lin50Model3, crossValData3);
lin50Results4 = svmclassify(lin50Model4, crossValData4);

%Test polynomial classifiers
poly50Results1 = svmclassify(poly50Model1, crossValData1);
poly50Results2 = svmclassify(poly50Model2, crossValData2);
poly50Results3 = svmclassify(poly50Model3, crossValData3);
poly50Results4 = svmclassify(poly50Model4, crossValData4);

%Test rbf classifiers
rbf50Results1 = svmclassify(rbf50Model1, crossValData1);
rbf50Results2 = svmclassify(rbf50Model2, crossValData2);
rbf50Results3 = svmclassify(rbf50Model3, crossValData3);
rbf50Results4 = svmclassify(rbf50Model4, crossValData4);

%print out average error rate
%if we add our guesses with the actual labels, the incorrect ones will = 0
%and the correct ones will either equal -2 or 2. So the percentage of 0s =
%error rate.

%Linear
guessedLabels = lin50Results1 + crossValLabels1;
averageErrorRateLinear = 1 - (nnz(guessedLabels) ./ size(crossValLabels1,1));

guessedLabels = lin50Results2 + crossValLabels2;
averageErrorRateLinear = averageErrorRateLinear + 1 - (nnz(guessedLabels) ./ size(crossValLabels2,1));

guessedLabels = lin50Results3 + crossValLabels3;
averageErrorRateLinear = averageErrorRateLinear + 1 - (nnz(guessedLabels) ./ size(crossValLabels3,1));

guessedLabels = lin50Results4 + crossValLabels4;
averageErrorRateLinear = averageErrorRateLinear + 1 - (nnz(guessedLabels) ./ size(crossValLabels4,1));

averageErrorRateLinear = averageErrorRateLinear ./ 4

%Polynomial
guessedLabels = poly50Results1 + crossValLabels1;
averageErrorRatePolynomial = 1 - (nnz(guessedLabels) ./ size(crossValLabels1,1));

guessedLabels = poly50Results2 + crossValLabels2;
averageErrorRatePolynomial = averageErrorRatePolynomial + 1 - (nnz(guessedLabels) ./ size(crossValLabels2,1));

guessedLabels = poly50Results3 + crossValLabels3;
averageErrorRatePolynomial = averageErrorRatePolynomial + 1 - (nnz(guessedLabels) ./ size(crossValLabels3,1));

guessedLabels = poly50Results4 + crossValLabels4;
averageErrorRatePolynomial = averageErrorRatePolynomial + 1 - (nnz(guessedLabels) ./ size(crossValLabels4,1));

averageErrorRatePolynomial = averageErrorRatePolynomial ./ 4

%Gaussian rbf
guessedLabels = rbf50Results1 + crossValLabels1;
averageErrorRateRbf = 1 - (nnz(guessedLabels) ./ size(crossValLabels1,1));

guessedLabels = rbf50Results2 + crossValLabels2;
averageErrorRateRbf = averageErrorRateRbf + 1 - (nnz(guessedLabels) ./ size(crossValLabels2,1));

guessedLabels = rbf50Results3 + crossValLabels3;
averageErrorRateRbf = averageErrorRateRbf + 1 - (nnz(guessedLabels) ./ size(crossValLabels3,1));

guessedLabels = rbf50Results4 + crossValLabels4;
averageErrorRateRbf = averageErrorRateRbf + 1 - (nnz(guessedLabels) ./ size(crossValLabels4,1));

averageErrorRateRbf = averageErrorRateRbf ./ 4