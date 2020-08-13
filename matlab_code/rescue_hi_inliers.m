function features_info = rescue_hi_inliers( filter, features_info, cam )

chi2inv_2_95 = 5.9915;
% chi_099_2 = 9.2103;

features_info = predict_camera_measurements( filter.x_k_k, cam, features_info );
features_info = calculate_derivatives( filter.x_k_k, cam, features_info );

for i=1:length(features_info)
    
    if (features_info(i).individually_compatible==1)&&(features_info(i).low_innovation_inlier==0)
        hi = features_info(i).h';
        Si = features_info(i).H*sparse(double(filter.p_k_k))*features_info(i).H';
        nui = features_info(i).z - hi;
        nui = sparse(double(nui));
        if nui'*inv(Si)*nui<chi2inv_2_95
            features_info(i).high_innovation_inlier=1;
        else
            features_info(i).high_innovation_inlier=0;
        end
    end
end