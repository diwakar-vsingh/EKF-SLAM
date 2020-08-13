function J = jacob_rpy2q(qini)

rpy = tr2rpy(q2tr(qini));

[X,FVAL,EXITFLAG,OUTPUT,J] = fsolve('F_test_rpy2q',rpy',...
                                           optimset('Display','off',...
                                           'NonlEqnAlgorithm','gn'),rpy');