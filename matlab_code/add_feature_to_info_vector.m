function features_info = add_feature_to_info_vector( uv, im_k, X_RES, features_info, step, newFeature, init_feature_descriptor)

half_patch_size_when_initialized = 20;
half_patch_size_when_matching = 6;
new_position = length(features_info)+1;

features_info(new_position).patch_when_initialized =...
    im_k(uv(2,1)-half_patch_size_when_initialized:uv(2,1)+half_patch_size_when_initialized,...
    uv(1,1)-half_patch_size_when_initialized:uv(1,1)+half_patch_size_when_initialized);
features_info(new_position).feature_when_initialized = init_feature_descriptor;
features_info(new_position).patch_when_matching = zeros( 2*half_patch_size_when_matching+1, 2*half_patch_size_when_matching+1 );
features_info(new_position).r_wc_when_initialized=X_RES(1:3);
features_info(new_position).R_wc_when_initialized=q2r(X_RES(4:7));
features_info(new_position).uv_when_initialized=uv';
features_info(new_position).half_patch_size_when_initialized=half_patch_size_when_initialized;
features_info(new_position).half_patch_size_when_matching=half_patch_size_when_matching;
features_info(new_position).times_predicted=0;
features_info(new_position).times_measured=0;
features_info(new_position).init_frame = step;
features_info(new_position).init_measurement = uv;
features_info(new_position).type = 'inversedepth';
features_info(new_position).yi = newFeature;
features_info(new_position).individually_compatible = 0;
features_info(new_position).low_innovation_inlier = 0;
features_info(new_position).high_innovation_inlier = 0;
features_info(new_position).z = [];
features_info(new_position).h = [];
features_info(new_position).H = [];
features_info(new_position).S = [];
features_info(new_position).state_size = 6;
features_info(new_position).measurement_size = 2;
features_info(new_position).R = eye(features_info(new_position).measurement_size);