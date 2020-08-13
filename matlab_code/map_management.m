function [ filter, features_info ] = map_management( filter,...
    features_info, cam, im,  min_number_of_features_in_image, step )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% delete features, if necessary
[ filter, features_info ] = delete_features( filter, features_info );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

measured = 0;
for i=1:length(features_info)
    if (features_info(i).low_innovation_inlier || features_info(i).high_innovation_inlier) measured = measured + 1; end
end

% update features info
features_info = update_features_info( features_info );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% convert features from inverse depth to cartesian, if necessary
[ filter, features_info ] = inversedepth_2_cartesian( filter, features_info );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% initialize features randomly
if measured == 0
    [ filter, features_info ] = initialize_features( step, cam,...
        filter, features_info, min_number_of_features_in_image, im );
else
    if measured < min_number_of_features_in_image
        [ filter, features_info ] = initialize_features( step, cam,...
            filter, features_info, min_number_of_features_in_image - measured, im );
    end
end