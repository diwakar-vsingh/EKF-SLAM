function plotUncertainSurfaceXZ( C, nu, chi2, color, randSphere6D, nPointsRand )

% jcivera, 12/1/06

if eig(C)>ones(size(C,1),1)*eps ~= ones(size(C,1),1)

    C = C + eps*diag(size(C,1));

end

K = chol( C )';

id_points = K*randSphere6D +...
    [ ones(1,nPointsRand)*nu(1); ones(1,nPointsRand)*nu(2); ones(1,nPointsRand)*nu(3);...
    ones(1,nPointsRand)*nu(4); ones(1,nPointsRand)*nu(5); ones(1,nPointsRand)*nu(6)];

rho_positive_points = id_points(6,:)>0;
id_points_rho_positive = id_points(:,rho_positive_points);

if ~isempty(id_points_rho_positive) && size(id_points_rho_positive,2)>10

    cartesian_points = inversedepth2cartesian( id_points_rho_positive );
    cartesian_points = double(cartesian_points);

    k = convhull( cartesian_points(1,:)', cartesian_points(3,:)' );
    plot3( cartesian_points(1, [k ; k(1)]), zeros(1,size(k,1)+1), cartesian_points(3, [k ; k(1)]),...
        'color', color, 'LineWidth', 1.5);
    hold on;

end