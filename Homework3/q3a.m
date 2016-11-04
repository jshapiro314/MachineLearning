%load in files for problem
load('./iris_1/test_data.mat')
load('./iris_1/test_labels.mat')
load('./iris_1/train_data.mat')
load('./iris_1/train_labels.mat')

%run svmtrain
classifier = svmtrain(train_data,train_labels,'showplot',true)

%run svmclassify
output = svmclassify(classifier, test_data, 'showplot', true)

%print out error rate
%if we add our guesses with the actual labels, the incorrect ones will = 0
%and the correct ones will either equal -2 or 2. So the percentage of 0s =
%error rate.

guessedLabels = output + test_labels;
errorRate = 1 - (nnz(guessedLabels) ./ size(test_labels,1))
