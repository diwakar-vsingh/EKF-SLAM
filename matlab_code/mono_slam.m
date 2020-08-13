%% Clear Workspace
clc
clear
close all

%% Control Random Number generator
rng default
rng(0);

%% -----------------------------------------------------------------------
% Sequence, camera and filter tuning parameters, variable initialization
%-----------------------------------------------------------------------

% Camera calibration
cam = initialize_cam;

% Set plot windows
set_plots;

% Sequence path and initial image
sequencePath = '../sequences/ic/rawoutput';
initIm = 1;
lastIm = 100; % 2169

% Initialize state vector and covariance
[x_k_k, p_k_k] = initialize_x_and_p;

% Initialize EKF filter
sigma_a             = 0.007; % standar deviation for linear acceleration noise
sigma_alpha         = 0.007; % standar deviation for angular acceleration noise
sigma_image_noise   = 1.0; % standar deviation for measurement noise
filter = ekf_filter( double(x_k_k), double(p_k_k), sigma_a, sigma_alpha, sigma_image_noise, 'constant_velocity' );

% variables initialization
features_info = [];
trajectory = zeros( 7, lastIm - initIm );

% other
min_number_of_features_in_image = 25;
generate_random_6D_sphere;
measurements = []; 
predicted_measurements = [];

%% ---------------------------------------------------------------
% Main loop
%---------------------------------------------------------------

im = takeImage( sequencePath, initIm );

for step=initIm+1:lastIm
    
    % Map management (adding and deleting features; and converting inverse depth to Euclidean)
    [ filter, features_info ] = map_management( filter, features_info, cam, im, min_number_of_features_in_image, step );

    % EKF prediction (state and measurement prediction)
    [ filter, features_info ] = ekf_prediction( filter, features_info );
    
    % Grab image
    im = takeImage( sequencePath, step );
    
    % Search for individually compatible matches
    features_info = search_IC_matches( filter, features_info, cam, im );
    
    % 1-Point RANSAC hypothesis and selection of low-innovation inliers
    features_info = ransac_hypotheses( filter, features_info, cam );
    
    % Partial update using low-innovation inliers
    filter = ekf_update_li_inliers( filter, features_info );
    
    % "Rescue" high-innovation inliers
    features_info = rescue_hi_inliers( filter, features_info, cam );
    
    % Partial update using high-innovation inliers
    filter = ekf_update_hi_inliers( filter, features_info );

    % Plots,
    plots; display( step );
    
    % Save images
    saveas( figure_all, sprintf( '%s/image%04d.fig', directory_storage_name, step ), 'fig' );

end

% Mount a video from saved Matlab figures
fig2avi;