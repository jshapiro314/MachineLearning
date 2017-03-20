%Load in the cleaned up test and train data
%MAY NEED TO MODIFY WHAT IS LOADED IN. THIS CAN ALSO BE DONE IN A DIFFERENT
%FILE.
load('trainMushroomData.mat')
load('trainMushroomLabels.mat')
load('testMushroomData.mat')
load('testMushroomLabels.mat')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%HERE ARE THE VARIABLES THAT NEED TO BE CHANGED. SIMPLY ASSIGN THE TRAINING
%AND TESTING VARIABLES TO THE CORRECT MATRICIES AND YOU DON'T HAVE TO
%MODIFY THE CODE BELOW. KEEP THE @ IN THE CLASSIFIER FUNCTION LINE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
trainData = trainGt;
trainLabel = trainLt;
testData = testG;
testLabel = testL;
classifierFunc = @SVM_class_func;

%Call the sequentialFeatureSelection function
%The SVM_class_func is located at the bottom of this document
[fs1,history] = sequentialfs(classifierFunc,trainData,trainLabel)

%Now that we have fs1, a vector of 1s and 0s, lets create a new test
%data matrix & training data matrix that only includes test & training data
%colums where fs1 is 1. We are essentially grabbing the feature vectors
%that sequentialfs told us were important.
testingData = [];
trainingData = [];
%fs1 = [0,0,0,0,0,1,0,0,1,1,0,0,1,1,0,0,0,0,0,1];
for i=1:numel(fs1)
    if fs1(i) == 1
        testingData = [testingData testData(:,i)];
        trainingData = [trainingData trainData(:,i)];
    end
end

%Lets calculate error with the SVM used in the original step
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


