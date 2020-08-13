function [features_info] = ransac_hypotheses( filter, features_info, cam )

p_at_least_one_spurious_free = 0.99; % default value
% RANSAC threshold should have a low value (less than the standard
% deviation of the filter measurement noise); as high innovation points
% will be later rescued
threshold = get_std_z(filter);

n_hyp = 1000; % initial number of iterations, will be updated
max_hypothesis_support = 0; % will be updated

[state_vector_pattern, z_id, z_euc] = generate_state_vector_pattern( features_info, get_x_k_km1(filter) );

for i = 1:n_hyp
    
    % select random match
    [zi,position,num_IC_matches] = select_random_match(features_info);
    
    % 1-match EKF state update
    x_k_km1 = get_x_k_km1(filter);
    p_k_km1 = get_p_k_km1(filter);
    hi = features_info(position).h';
    Hi = sparse(features_info(position).H);
    S = full(Hi*p_k_km1*Hi' + features_info(position).R);
    K = p_k_km1*Hi'*inv(S);
    xi = x_k_km1 + double(K*( zi - hi ));
    
    % predict features
    % Compute hypothesis support: predict measurements and count matches
    % under a threshold
    % features_info_i = predict_camera_measurements( xi, cam, features_info );
    % hypothesis_support = count_matches_under_a_threshold( features_info_i );
    % Fast version of computing hypothesis support, avoiding loops with
    % high cost in Matlab
    [hypothesis_support, positions_li_inliers_id, positions_li_inliers_euc] = compute_hypothesis_support_fast( xi, cam, state_vector_pattern, z_id, z_euc, threshold );
    
    if hypothesis_support > max_hypothesis_support
        max_hypothesis_support = hypothesis_support;
        features_info = set_as_most_supported_hypothesis( features_info, positions_li_inliers_id, positions_li_inliers_euc );
        epsilon = 1-(hypothesis_support/num_IC_matches);
        n_hyp = ceil(log(1-p_at_least_one_spurious_free)/log(1-(1-epsilon)));
        if n_hyp==0 break; end
    end
    
    if i>n_hyp break; end
    
end