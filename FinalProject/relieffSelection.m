%Load in the cleaned up test and train data
%MAY NEED TO MODIFY WHAT IS LOADED IN. THIS CAN ALSO BE DONE IN A DIFFERENT
%FILE.
load('trainVoiceData.mat')
load('trainVoiceLabels.mat')
load('testVoiceData.mat')
load('testVoiceLabels.mat')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%HERE ARE THE VARIABLES THAT NEED TO BE CHANGED. SIMPLY ASSIGN THE TRAINING
%AND TESTING VARIABLES TO THE CORRECT MATRICIES AND YOU DON'T HAVE TO
%MODIFY THE CODE BELOW. KEEP THE @ IN THE CLASSIFIER FUNCTION LINE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
trainData = trainGt;
trainLabel = trainLt;
testData = testG;
testLabel = testL;
kVal = 10;
featureNum = 3;
classifierFunc = @SVM_class_func;


%Call the relieff function. Experiment with K values
[ranked,weight] = relieff(trainData,trainLabel,kVal)

%Now that we have ranked & weight, lets create a new test
%data matrix & training data matrix that only includes up to 80% of the sum
% of the weight. We are essentially grabbing the feature vectors
%that relieff told us were important. We can change the number of columns
testingData = [];
trainingData = [];
sum = 0;
for i=1:featureNum
        testingData = [testingData testData(:,ranked(i))];
        trainingData = [trainingData trainData(:,ranked(i))];
end

%Lets calculate error with the SVM
numWrong = classifierFunc(trainingData,trainLabel,testingData,testLabel);
numWrong/size(testLabel,1)*100

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%PUT ALL CLASSIFIER FUNCTIONS HERE. MAKE SURE THEY TAKE THE FORM OF INPUT
%SHOWN BELOW. THIS IS ESSENTIAL FOR THE FEATURE SELECTION TO WORK
%Create the error calculation function (where the classifier goes!) --
%currently using linear svm from voice.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function err = SVM_class_func(xTrain,yTrain,xTest,yTest)
    model = svmtrain(xTrain,yTrain,'kernel_function','linear','boxconstraint',1);
    err = sum(svmclassify(model,xTest) ~= yTest);
end


