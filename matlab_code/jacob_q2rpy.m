function J = jacob_q2rpy(q)

% q = tr2q(rpy2tr(rpy(1),rpy(2),rpy(3)));

[X,FVAL,EXITFLAG,OUTPUT,J] = fsolve('F_test_q2rpy',q',...
                                           optimset('Display','off',...
                                           'NonlEqnAlgorithm','gn'),q');