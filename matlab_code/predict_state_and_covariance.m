function [ X_km1_k, P_km1_k ] = predict_state_and_covariance( X_k, P_k, type, SD_A_component_filter, SD_alpha_component_filter )

X_k = double(X_k);
P_k = double(P_k);
delta_t = 1;

% camera motion prediction
Xv_km1_k = fv( X_k(1:13,:), delta_t, type, SD_A_component_filter, SD_alpha_component_filter  );

% features prediction
X_km1_k = [ Xv_km1_k; X_k( 14:end,: ) ];

% state transition equation derivatives
F = sparse( dfv_by_dxv( X_k(1:13,:),zeros(6,1),delta_t, type ) );

% state noise
linear_acceleration_noise_covariance = (SD_A_component_filter*delta_t)^2;
angular_acceleration_noise_covariance = (SD_alpha_component_filter*delta_t)^2;
Pn = sparse (diag( [linear_acceleration_noise_covariance linear_acceleration_noise_covariance linear_acceleration_noise_covariance...
        angular_acceleration_noise_covariance angular_acceleration_noise_covariance angular_acceleration_noise_covariance] ) );

Q = func_Q( X_k(1:13,:), zeros(6,1), Pn, delta_t, type);

size_P_k = size(P_k,1);

P_km1_k = [ F*P_k(1:13,1:13)*F' + Q         F*P_k(1:13,14:size_P_k);
            P_k(14:size_P_k,1:13)*F'        P_k(14:size_P_k,14:size_P_k)];