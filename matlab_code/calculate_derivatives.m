function features_info = calculate_derivatives( x_k_km1, cam, features_info )

x_v = x_k_km1(1:13);
x_features = x_k_km1(14:end);

for i=1:length(features_info)
    
    if ~isempty(features_info(i).h)
        if strcmp(features_info(i).type, 'cartesian')
            y = x_features(1:3);
            x_features = x_features(4:end);
            features_info(i).H = sparse(calculate_Hi_cartesian( x_v, y, cam, i, features_info ));
        else
            y = x_features(1:6);
            x_features = x_features(7:end);
            features_info(i).H = sparse(calculate_Hi_inverse_depth( x_v, y, cam, i, features_info ));
        end
        
    else
        if strcmp(features_info(i).type, 'cartesian')
            x_features = x_features(4:end);
        end
        if strcmp(features_info(i).type, 'inversedepth')
            x_features = x_features(7:end);
        end
    end
    
end