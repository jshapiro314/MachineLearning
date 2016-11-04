% the function plots a mixture of two 2D gaussians
% Input : a struct called mixture, with the following fields
%       mu1     : mean of Gaussian 1
%       sigma1  : covariance matrix of Gaussian 1
%       mu2     : mean of Gaussian 2
%       sigma2  : covariance matrix of Gaussian 2
%       wts     : mixture weights for each Gaussian (default = 0.5)
function plotMixtureGaussians( mixture )

    % determine the contour described by the mixture
    m1 = mixture.mu1';
    s1 = mixture.sigma1;
    m2 = mixture.mu2';
    s2 = mixture.sigma2;
    w1 = mixture.wts(1);
    w2 = mixture.wts(2);
    
    xlim = -5:0.1:5;
    ylim = -5:0.1:5;
    [x,y] = meshgrid( xlim, ylim );
    
    z = zeros( length(xlim), length(ylim) );
    for ii = 1:length(xlim)
        for jj = 1:length(ylim)
            X = [x(ii,jj), y(ii,jj)]';
            t1 = w1 * gaussian2D( X, m1, s1 );
            t2 = w2 * gaussian2D( X, m2, s2 );
            if t1 > t2
                z(ii,jj) = 1;
            else 
                z(ii,jj) = -1;
            end
        end
    end
    
    contour(x,y,z, 'XData', xlim, 'YData', ylim)
    hold on;
    
    % plot the means
    x = [ mixture.mu1(1) mixture.mu2(1)];
    y = [ mixture.mu1(2) mixture.mu2(2)];
    plot( x, y, 'r.', 'Markersize', 5);
    
    % plot the individual gaussians
    plotGaussian( m1, s1, xlim, ylim );
    plotGaussian( m2, s2, xlim, ylim );
end

function p = gaussian2D( X, mu, s )
    p = (1/( 2 *pi *det(s)^0.5)) * exp(-0.5 * (X - mu)' * (s \ (X-mu)) );
end

function plotGaussian( mu, s, xlim, ylim )

    hold on
    [x,y] = meshgrid( xlim, ylim );
    
    z = zeros( length(xlim), length(ylim) );
    for ii = 1:length(xlim)
        for jj = 1:length(ylim)
            X = [x(ii,jj), y(ii,jj)]';
            z(ii,jj) = gaussian2D( X, mu, s ); 
        end
    end
    contour(x,y,z, 'XData', xlim, 'YData', ylim);
end