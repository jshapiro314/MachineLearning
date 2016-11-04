function [ output ] = classifyGaussians( test_data, gaussians )
%CLASSIFYGAUSSIANS given the parameters for 2 gaussians, classify a given
%point to a specific gaussian
%   Detailed explanation goes here
output = zeros(size(test_data,1),1)

detCov1 = gaussians(1,4) * gaussians(1,5);
detCov2 = gaussians(2,4) * gaussians(2,5);
invCov1 = [1./gaussians(1,4),1./gaussians(1,5)];
invCov2 = [1./gaussians(2,4),1./gaussians(2,5)];
mean1 = [gaussians(1,2),gaussians(1,3)];
mean2 = [gaussians(2,2),gaussians(2,3)];
weight1 = gaussians(1,1);
weight2 = gaussians(2,1);

for i=1:size(output)

    gauss1Val = 1/((2*pi)^(2/2)*detCov1^(1/2))*exp((-1/2)*sum(invCov1.*((test_data(i)-mean1).^2)))*weight1;
    gauss2Val = 1/((2*pi)^(2/2)*detCov2^(1/2))*exp((-1/2)*sum(invCov2.*((test_data(i)-mean2).^2)))*weight2;
    
    probX = [gauss1Val,gauss2Val] * [weight1;weight2];
    
    temp = [weight1,weight2];
    temp2 = [probX,probX];
    probYX = [gauss1Val,gauss2Val].*temp./temp2;
    
    
    if(probYX(1) > probYX(2))
        output(i) = 1;
    else
        output(i) = -1;
    end

end

