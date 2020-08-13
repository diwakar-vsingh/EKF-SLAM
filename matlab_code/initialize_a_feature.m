function [ filter, features_info, uv ] = initialize_a_feature( step, cam, img, filter, features_info )

% numerical values
half_patch_size_when_initialized = 20;
half_patch_size_when_matching = 6;
excluded_band = half_patch_size_when_initialized + 1;
max_initialization_attempts = 1;
initializing_box_size = [60;40];
initializing_box_semisize = initializing_box_size/2;
initial_rho = 1;
std_rho = 1;
std_pxl = get_std_z(filter);

features_info = predict_camera_measurements( get_x_k_k(filter), cam, features_info );

uv_pred = [];
for i=1:length(features_info)
    uv_pred = [uv_pred features_info(i).h'];
end


search_region_center = rand(2,1);
search_region_center(1) = round(search_region_center(1)*(cam.nCols-2*excluded_band-2*initializing_box_semisize(1)))...
    +excluded_band+initializing_box_semisize(1);
search_region_center(2) = round(search_region_center(2)*(cam.nRows-2*excluded_band-2*initializing_box_semisize(2)))...
    +excluded_band+initializing_box_semisize(2);

% Extract FAST corners
corners = detectFASTFeatures(img, 'ROI', [search_region_center(1)-initializing_box_semisize(1),...
                                          search_region_center(2)-initializing_box_semisize(2),...
                                          initializing_box_size(1), initializing_box_size(2)], 'MinContrast', 0.40);
all_uv = corners.Location';

% Are there corners in the box?
are_there_corners = not(isempty(all_uv));

% Are there other features in the box?
if ~isempty(uv_pred)
    total_features_number = size(uv_pred,2);
    features_in_the_box =...
        (uv_pred(1,:)>ones(1,total_features_number)*(search_region_center(1)-initializing_box_semisize(1)))&...
        (uv_pred(1,:)<ones(1,total_features_number)*(search_region_center(1)+initializing_box_semisize(1)))&...
        (uv_pred(2,:)>ones(1,total_features_number)*(search_region_center(2)-initializing_box_semisize(2)))&...
        (uv_pred(2,:)<ones(1,total_features_number)*(search_region_center(2)+initializing_box_semisize(2)));
    are_there_features = (sum(features_in_the_box)~=0);
else
    are_there_features = false;
end

if(are_there_corners&&(~are_there_features))
    [ext_features, valid_points]    = extractFeatures(img, corners, 'Method', 'FREAK');
    idx                             = ismember(valid_points.Location, valid_points.selectStrongest(1).Location, 'rows');
    init_feature_descriptor         = binaryFeatures(ext_features.Features(idx,:));
    uv                              = double(valid_points.Location(idx, :));
else
    uv=[];
end
uv = uv';

if(~isempty(uv))
    
    % add the feature to the filter
    [ X_RES, P_RES, newFeature ] = add_features_inverse_depth( uv, get_x_k_k(filter),...
        get_p_k_k(filter), cam, std_pxl, initial_rho, std_rho );
    filter = set_x_k_k(filter, X_RES);
    filter = set_p_k_k(filter, P_RES);
    
    % add the feature to the features_info vector
    features_info = add_feature_to_info_vector( uv, img, X_RES, features_info, step, newFeature, init_feature_descriptor );
    
end

for i=1:length(features_info)
    features_info(i).h = [];
end
