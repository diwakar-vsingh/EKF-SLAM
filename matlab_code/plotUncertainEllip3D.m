function plotUncertainEllip3D( C, nu, chi2, color, alpha )

[ X, Y, Z ] = sphere;

nPoints = size(X,2)*size(X,1);

X = reshape( X, 1, nPoints )*sqrt(chi2);
Y = reshape( Y, 1, nPoints )*sqrt(chi2);
Z = reshape( Z, 1, nPoints )*sqrt(chi2);

K = chol( C )';

XYZprima = K*[ X; Y; Z ] + [ ones(1,nPoints)*nu(1); ones(1,nPoints)*nu(2); ones(1,nPoints)*nu(3)];

X = reshape( XYZprima(1,:), sqrt(nPoints), sqrt(nPoints) );
Y = reshape( XYZprima(2,:), sqrt(nPoints), sqrt(nPoints) );
Z = reshape( XYZprima(3,:), sqrt(nPoints), sqrt(nPoints) );

if alpha == 1
    mesh( X, Y, Z, 'FaceColor', color, 'EdgeColor', color, 'LineWidth', 1.5 );
else
    mesh( X, Y, Z, 'FaceColor', color , 'FaceAlpha', 0.2, 'EdgeColor', color, 'EdgeAlpha', 0.4);
end
hold on;