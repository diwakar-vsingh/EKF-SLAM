function [ X_RES, P_RES, newFeature ] = add_features_inverse_depth( uvd, X, P, cam, std_pxl, initial_rho, std_rho )

nNewFeat = size( uvd, 2 );

if nNewFeat == 0
    X_RES = X;
    P_RES = P;
    return;

else

    % Camera state
    Xv = X(1:13);

    X_RES = X;
    P_RES = P;

    for i = 1:nNewFeat
        newFeature = hinv( uvd(:,i), Xv, cam, initial_rho );
        X_RES = [ X_RES; newFeature ];
        P_RES = add_a_feature_covariance_inverse_depth( P_RES, uvd(:,i), Xv, std_pxl, std_rho, cam );
    end

end