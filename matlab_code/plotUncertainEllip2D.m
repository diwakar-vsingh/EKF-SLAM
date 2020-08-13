function plotUncertainEllip2D(C,nu,chi2,color,linewidth)

hold on;

th=0:2*pi/100:2*pi;

x=[cos(th') sin(th')]'*sqrt(chi2);

if(min(eig(C))<0)
    C=eye(2);
    color=[0 0 0];
    fprintf('NPSD matrix, a black false ellipse has been plot\n');
end
K=chol(C)';

nPoints=size(th,2);
y=K*x+[ones(1,nPoints)*nu(1);ones(1,nPoints)*nu(2)];

h = plot(y(1,:),y(2,:));

set(h,'Color',color,'LineWidth',linewidth);