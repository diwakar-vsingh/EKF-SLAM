function features_info = search_IC_matches( filter, features_info, cam, im)

% Predict features and individual search regions
features_info = predict_camera_measurements( get_x_k_km1(filter), cam, features_info );
features_info = calculate_derivatives( get_x_k_km1(filter), cam, features_info );
for i=1:length(features_info)
    if ~isempty(features_info(i).h)
        features_info(i).S = features_info(i).H*get_p_k_km1(filter)*features_info(i).H' + features_info(i).R;
    end
end

% Warp patches according to predicted motion
% features_info = predict_features_appearance( features_info, get_x_k_km1(filter), cam );

% Find correspondences in the search regions using normalized
% cross-correlation
features_info = matching( im, features_info, cam );