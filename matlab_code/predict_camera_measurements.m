function features_info = predict_camera_measurements( x_k_k, cam, features_info )

% Pinhole model
t_wc = x_k_k(1:3);
r_wc = q2r(x_k_k(4:7));
features = x_k_k( 14:end );

for i = 1:length(features_info)
    
    if strcmp(features_info(i).type, 'cartesian')
        yi = features( 1:3 );
        features = features( 4:end );
        hi = hi_cartesian( yi, t_wc, r_wc, cam, features_info(i) );
        if (~isempty(hi))
            features_info(i).h = hi';
        end
    end
    
    if strcmp(features_info(i).type, 'inversedepth')
        yi = features( 1:6 );
        features = features( 7:end );
        hi = hi_inverse_depth( yi, t_wc, r_wc, cam, features_info(i) );
        if (~isempty(hi))
            features_info(i).h = hi';
        end
    end
    
end