function plot_points_and_classifier(X,y,theta)
%  plot_points_and_classifier(X,y,theta)
%
%    plot data and decision boundary.
%    X is a n x 2 matrix, y is a n x 1 column vector.
%    theta is a 2 x 1 vector returned by the Perceptron training
%    algorithm.
%    y=+1 points are plotted as red dots, while y=-1 points are plotted
%    as
%    blue 'x's. The decision boundary is plotted as a black line.

    plot(X(y==1,1), X(y==1,2),'r.');
    hold on;
    plot(X(y==-1,1), X(y==-1,2),'bx');

    if( length(theta) == 2 )
    v = axis;
    xmax = v(2);
    ymax2 = -xmax*theta(1)/theta(2);
    plot([0, xmax], [0, ymax2],'k','Linewidth',3);
    hold off;
    
    else
        draw_line(theta);
    end
    

end

function draw_line( theta )
% first find one intercept
if theta(2) ~= 0
    p1(1) = 4;%(- theta(3))/theta(1);
    p1(2) = -( theta(3) + theta(1) * p1(1) )/theta(2);
    
    p2(1) = -4;
    p2(2) = -( theta(3) + theta(1) * p2(1) )/theta(2);
else
    p1(1) = 0;
    p1(2) = -theta(3)/theta(2);
    
    p2(1) = 0;
    p2(2) = p1(2) + 10;
end


% finally draw the line
line( [p1(1), p2(1)], [p1(2), p2(2)] );
end