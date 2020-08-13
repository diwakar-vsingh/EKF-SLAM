function features_info = matching(img, features_info, cam)
chi_095_2 = 5.9915;
% chi_099_2 = 9.2103;

for idx_feature = 1:length(features_info) % for every feature in the map
    
    if ~isempty(features_info(idx_feature).h) % if it is predicted, search in the region
        
        % initial feature descriptor corresponding to the current feature
        init_feature_descriptor = features_info(idx_feature).feature_when_initialized;
        
        % Predicted image location
        h   = features_info(idx_feature).h;
        S   = features_info(idx_feature).S;
        
        if eig(S) < 100 % if the ellipse is too big, i.e. uncertainity is huge, do not search
            
            % Predict search region in the 95% probability region
            invS = inv(S);
            
            half_search_region_size_x = ceil(2*sqrt(S(1,1)));
            half_search_region_size_y = ceil(2*sqrt(S(2,2)));
            
            row1 = max(1, round(h(2) - half_search_region_size_y));
            row2 = min(round(h(2) + half_search_region_size_y), cam.nRows);
            col1 = max(1, round(h(1) - half_search_region_size_x));
            col2 = min(round(h(1) + half_search_region_size_x), cam.nCols);

            corners     = detectFASTFeatures(img, 'ROI', [col1, row1, col2-col1, row2-row1]);
            n           = length(corners);
            location    = [];
            metric      = [];
            for j=1:n
                hp      = corners.Location(j,:);
                u       = hp(1);
                v       = hp(2);
                nu      = [u-h(1); v-h(2)];
                if nu'*invS*nu < chi_095_2
                    location    = [location; corners.Location(j,:)];
                    metric      = [metric; corners.Metric(j)];
                end
            end
            if (~isempty(location))
                points      = cornerPoints(location, 'Metric', metric);
                [predicted_feature_descriptor, valid_corners] = extractFeatures(img, points, 'Method', 'FREAK');
                indexPairs  = matchFeatures(init_feature_descriptor, predicted_feature_descriptor, 'Method', 'Approximate',...
                    'MaxRatio', 1, 'Unique', true, 'MatchThreshold', 100);
                
                % Retrieve the locations of the corresponding points for each image.
                matchedPoints = valid_corners(indexPairs(:,2),:);
                if (~isempty(matchedPoints))
                    features_info(idx_feature).individually_compatible = 1;
                    features_info(idx_feature).z = matchedPoints.Location';
                end
            end
        end
    end
end


