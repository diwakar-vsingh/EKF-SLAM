function filter = ekf_update_hi_inliers( filter, features_info )

% mount vectors and matrices for the update
z = [];
h = [];
H = [];

for i=1:length( features_info )
    
    if features_info(i).high_innovation_inlier == 1
        z = [z; features_info(i).z(1); features_info(i).z(2)];
        h = [h; features_info(i).h(1); features_info(i).h(2)];
        H = [H; features_info(i).H];
    end
    
end

R = eye(length(z));

[ filter.x_k_k, filter.p_k_k ] = update( filter.x_k_k, filter.p_k_k, H,...
    R, z, h );