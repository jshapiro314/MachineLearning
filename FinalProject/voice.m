
fileID = fopen('voice.csv');
formatSpec = '%s';
C = textscan(fileID,formatSpec);
row = size(C{1},1);
col = 21;


% G is the data matrix
G = zeros(row,col-1);

% L is the label vector, 1 for male, -1 for female
L = zeros(row, 1);

for i = 1:row
    a = strread(C{1}{i}, '%s', 'delimiter', ',');
    for j = 1:col-1
       G(i,j) = str2double(a{j,1});
    end
    label = (a{col,1});
    if strcmp(label, 'male') == 1
        L(i,1) = 1;
    else L(i,1) = -1;
    end
end

% shuffle rows since they are grouped as all male-labeled data followed by 
% all female-labeled data
X = horzcat(G,L);
shuffledData = X(randperm(size(X,1)),:);
G = shuffledData(:,1:col-1);
L = shuffledData(:,col);

% separate into training and test data 80/20
c = ceil(0.8*row);
trainGt = G(1:c,:);
testG = G(c+1:row,:);
trainLt = L(1:c,:);
testL = L(c+1:row,:);

% separate validation set 80/20 of training
s = size(trainGt,1);
c = ceil(0.8*s);
valG = trainGt(c+1:s,:);
valL = trainLt(c+1:s,:);
trainG = trainGt(1:c,:);
trainL = trainLt(1:c,:);


%======================== Kernel SVM ======================================

% svmLinear = svmtrain(trainG,trainL,'kernel_function','mlp', 'boxconstraint',0.5);
% svmLinearLabels = svmclassify(svmLinear,valG);
% err = testErrorSVM(valL, svmLinearLabels)

% svmLinear = svmtrain(trainG,trainL,'kernel_function','linear', 'boxconstraint',0.5);
% svmLinearLabels = svmclassify(svmLinear,valG);
% err = testErrorSVM(valL, svmLinearLabels)


%======================== KNN =============================================
% 
% [IDX,D] = knnsearch(trainG,valG, 'Distance', 'seuclidean','K',1);
% err = testErrorKNN(IDX,trainL,valL)

%=======================Naive Bayes========================================

% [prediction,class,errorRate]= NB(trainGt,trainLt,testG,testL);
% errorRate

%=======================Lora's Kernel Perceptron===========================

% alpha = kernel_perceptron(trainGt, trainLt, 'polynomial');
% n = size(trainGt,1);
% N = size(testG,1);
% test_labels_Perceptron = zeros(N,1);
% err = 0;
% 
% for i = 1:N
%     sum = 0;
%     for j = 1:n
%         sum = sum + alpha(j)*trainLt(j)*dot(trainGt(j,:), testG(i,:));
%     end
%     if sum > 0
%         test_labels_Perceptron(i) = 1;
%     else test_labels_Perceptron(i) = -1;    
%     end
%     if test_labels_Perceptron(i)~= testL(i)
%         err = err + 1;        
%     end
% end
% errRate = err./N

%======================Kernel Perceptron===================================
err = TestPerceptron( trainGt, trainLt, testG, testL, 'rbf', 1 );
err = err/size(testG,1)
