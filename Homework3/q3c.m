%load in files for problem
load('./iris_3/test_data.mat')
load('./iris_3/test_labels.mat')
load('./iris_3/train_data.mat')
load('./iris_3/train_labels.mat')

%create anonymous functions for each kernel we are testing
linearKernel = @(x_1,x_2) dot(x_1,x_2);
polynomialKernel = @(x_1,x_2) (1 + dot(x_1,x_2)).^3;
radialKernel = @(x_1,x_2) exp(-1 * norm(x_1-x_2).^2);

%call perceptron functions
outputLinear = kernelPerceptron(train_data,train_labels,test_data,test_labels,linearKernel);
outputPoly = kernelPerceptron(train_data,train_labels,test_data,test_labels,polynomialKernel);
outputRadial = kernelPerceptron(train_data,train_labels,test_data,test_labels,radialKernel);

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
%theta = perceptron_train(train_data,train_labels)
%errorRateLinear2 = perceptron_test(theta(1), test_data, test_labels)
