function features_info = update_features_info( features_info )

% reset features_info parameters
for i=1:length(features_info)
    if ~isempty(features_info(i).h)
        features_info(i).times_predicted = features_info(i).times_predicted + 1;
    end
    if (features_info(i).low_innovation_inlier||features_info(i).high_innovation_inlier)
        features_info(i).times_measured = features_info(i).times_measured + 1;
    end
    features_info(i).individually_compatible = 0;
    features_info(i).low_innovation_inlier = 0;
    features_info(i).high_innovation_inlier = 0;
    features_info(i).h = [];
    features_info(i).z = [];
    features_info(i).H = [];
    features_info(i).S = [];
end