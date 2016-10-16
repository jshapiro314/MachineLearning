

%load necessary data
load('SPECTtrainData.mat');
load('SPECTtrainLabels.mat');

%creating models from training data
%add 1 to every total ( + n~p where n = 2 & ~p = 0.5)
p1 = ones(1,22);
p0 = ones(1,22);
%get total of each feature for each group
for i=1:187
    if trainLabels(i) == 1
        p1 = p1 + trainData(i,:);
    else
        p0 = p0 + trainData(i,:);
    end
end

%and then divide by total number of points in each group + n (n=2)
counts = tabulate(trainLabels);
%also fetch probability of groups over total
P1Group = counts(2,3)/100;
P0Group = counts(1,3)/100;

counts(:,2) = counts(:,2) + 2;
p1 = p1 / counts(2,2);
p0 = p0 / counts(1,2);

%p1 & p0 are now arrays of the probabilities for each feature in each
%model. We use them to caluclate the probability of new datapoints, and
%based on the larger probability, we know how to classify them
%test data using models
load('SPECTtestLabels.mat');
load('SPECTtestData.mat');
guessedLabels = zeros(80,1);
tempP1 = P1Group;
tempP0 = P0Group;

for i=1:80
    for j=1:22
        tempP1 = tempP1 * p1(j)^(testData(i,j)) * (1-p1(j))^(1-testData(i,j));
        tempP0 = tempP0 * p0(j)^(testData(i,j)) * (1-p0(j))^(1-testData(i,j));
    end
    %disp(tempP1)
    %disp(tempP0)
    if tempP1 > tempP0
        guessedLabels(i) = 1;
    else
        guessedLabels(i) = 0;
    end
    
    tempP1 = P1Group;
    tempP0 = P0Group;
end

%print out error rate
%if we add our guesses with the actual labels, the incorrect ones will = 1
%and the correct ones will either equal 0 or 2. So the percentage of 1s =
%error rate.

guessedLabels = guessedLabels + testLabels;
errorTable = tabulate(guessedLabels)
errorRate = errorTable(2,3)

