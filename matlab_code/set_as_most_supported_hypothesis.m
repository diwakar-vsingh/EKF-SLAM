function features_info = set_as_most_supported_hypothesis( features_info, positions_li_inliers_id, positions_li_inliers_euc )

j_id = 1;
j_euc = 1;

for i=1:length(features_info)
    
    if ~isempty(features_info(i).z)
        if strcmp(features_info(i).type, 'cartesian')
            if positions_li_inliers_euc(j_euc)
                features_info(i).low_innovation_inlier = 1;
            else
                features_info(i).low_innovation_inlier = 0;
            end
            j_euc = j_euc + 1;
        end
        if strcmp(features_info(i).type, 'inversedepth')
            if positions_li_inliers_id(j_id)
                features_info(i).low_innovation_inlier = 1;
            else
                features_info(i).low_innovation_inlier = 0;
            end
            j_id = j_id + 1;
        end
    end
    
end