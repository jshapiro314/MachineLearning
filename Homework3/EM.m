function [param,loglikes]=EM(X,numClusters)
[numData,numFeature]=size(X);
numGauss = numClusters;
%initialize parameters for numGauss number of Gaussians
%each set of parameters contains the mixing proportions,means,and
%covariance for each Gaussian

%initialize P(eachGaussian), that is, the mixing proportion
mixProp = ones(numGauss,1)/numGauss;
mean = ones(numGauss,numFeature);
%this model assumes zero for all off-diagonal entries
covar = ones(numGauss,numFeature);
detCov = ones(numGauss,1);
invCov = ones(numGauss,numFeature);
for i = 1:numGauss
    detCov(i) = prod(covar(i,:));
    invCov(i,:) = 1./covar(i,:);
    %initialize means for each Gaussian
    mean(i,:)=X(randi(numData),:);
end

%number of effective datapoint in each Gaussian
N = zeros(numGauss,1);


%vector that stores P(x|y) for each x on each Gaussian
probXY = zeros(numData,numGauss);
probX = zeros(numData,1);


count = 0;
max = 1000;
loglikes = zeros(max,1);
%while not number of iterations < max, do
while count < max
    count = count + 1;
    
    %E-step
    
    %Use the full Gaussian model to compute P(X_i|y_i)
    for j = 1:numData
        for k = 1:numGauss
            
            probXY(j,k) = 1/((2*pi)^(numFeature/2)*detCov(k)^(1/2))*exp((-1/2)*sum(invCov(k,:).*((X(j,:)-mean(k,:)).^2)));
        end
        probX(j) = probXY(j,:)*mixProp;
    end
    temp = repmat(mixProp',numData,1);
    temp2 = repmat(probX,1,numGauss);
    probYX = probXY.*temp./temp2;
    %row vector of the sums of each column of probYX
    N = sum(probYX,1);
    
    %M-step
    for l = 1:numGauss
        sumMean = zeros(numGauss,numFeature);
        for m = 1:numData
            sumMean(l,:)= sumMean(l,:)+probYX(m,l)*X(m,:);
        end
        mean(l,:)=sumMean(l,:)/N(l);
        sumVar = zeros(numGauss,numFeature);
        for n = 1:numData
            sumVar(l,:)= sumVar(l,:)+(probYX(n,l)*(X(n,:)-mean(l,:)).^2);
        end
        covar(l,:)=sumVar(l,:)/N(l);
        invCov(l,:) = 1./covar(l,:);
        detCov(l) = prod(covar(l,:));
        mixProp(l)=N(l)/sum(N);
    end
    
    
    temp3 = repmat(mixProp',numData,1);
    p = probXY.*temp3;
    probX = sum(p,2);
    loglikes(count) = sum(log(probX));
end


%figure; plot( loglikes(1:100) );
%xlabel('Number of iterations');
%ylabel('Expected loglikelihood');
%title( ['k = ' num2str(numClusters)] );
%return the parameters and compute the log likelihood of the mixture models
param = horzcat(mixProp, mean, covar);
