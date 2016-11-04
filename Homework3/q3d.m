%load in files for problem
load('./iris_3/test_data.mat')
load('./iris_3/test_labels.mat')
load('./iris_3/train_data.mat')
load('./iris_3/train_labels.mat')

%calculate gaussians from EM algorithm
outputEM = EM(train_data,2);

%classify data based on gaussians
outputGaussians = classifyGaussians(test_data, outputEM);

%run svmtrain for linear classifier
classifier = svmtrain(train_data,train_labels,'boxconstraint',1)

%run svmclassify for linear classifier
outputLinear = svmclassify(classifier, test_data);

%run svmtrain for polynomial (3) classifier
classifier = svmtrain(train_data,train_labels,'kernel_function','polynomial','boxconstraint',1)

%run svmclassify for polynomial (3) classifier
outputPoly = svmclassify(classifier, test_data);

%run svmtrain for gaussian radial basis classifier
classifier = svmtrain(train_data,train_labels,'kernel_function','rbf','boxconstraint',1)

%run svmclassify for gaussian radial basis classifier
outputRadial = svmclassify(classifier, test_data);

%print out error rate
%if we add our guesses with the actual labels, the incorrect ones will = 0
%and the correct ones will either equal -2 or 2. So the percentage of 0s =
%error rate.

guessedLabels = outputLinear + test_labels;
errorRateLinear = 1 - (nnz(guessedLabels) ./ size(test_labels,1))

guessedLabels = outputPoly + test_labels;
errorRatePolynomial = 1 - (nnz(guessedLabels) ./ size(test_labels,1))

guessedLabels = outputRadial + test_labels;
errorRateRadial = 1 - (nnz(guessedLabels) ./ size(test_labels,1))

guessedLabels = outputGaussians + test_labels;
errorRateGaussian = 1 - (nnz(guessedLabels) ./ size(test_labels,1))

