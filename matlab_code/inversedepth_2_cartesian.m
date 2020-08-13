function [ filter, features_info ] = inversedepth_2_cartesian( filter, features_info )

linearity_index_threshold = 0.1;
X = get_x_k_k(filter);
P = get_p_k_k(filter);
X = double(X);
P = double(P);

for i=1:length(features_info)

    if strcmp(features_info(i).type, 'inversedepth')
        initialPositionOfFeature = 14;
        for j=1:i-1
            if strcmp(features_info(j).type, 'cartesian')initialPositionOfFeature = initialPositionOfFeature + 3; end
            if strcmp(features_info(j).type, 'inversedepth') initialPositionOfFeature = initialPositionOfFeature + 6; end
        end
        std_rho = sqrt(P(initialPositionOfFeature + 5,initialPositionOfFeature + 5));
        rho = X(initialPositionOfFeature + 5);
        std_d = std_rho/(rho^2);
        % camera 2 to point distance
        theta = X(initialPositionOfFeature + 3);
        phi = X(initialPositionOfFeature + 4);
        mi = m(theta,phi);
        x_c1 = X(initialPositionOfFeature:initialPositionOfFeature+2);
        x_c2 = X(1:3);
        p = inversedepth2cartesian( X(initialPositionOfFeature:initialPositionOfFeature+5) );
        d_c2p = norm(p - x_c2);
        % alpha (parallax)
        cos_alpha = ((p-x_c1)'*(p-x_c2))/(norm(p-x_c1)*norm(p-x_c2));

        % Linearity index
        linearity_index = 4*std_d*cos_alpha/d_c2p;

        if linearity_index<linearity_index_threshold
            size_X_old = size(X,1);
            % State Vector
            X = [X(1:initialPositionOfFeature-1); p; X(initialPositionOfFeature+6:end)];
            % Covariance
            dm_dtheta = [cos(phi)*cos(theta)  0   -cos(phi)*sin(theta)]';
            dm_dphi = [-sin(phi)*sin(theta)  -cos(phi)   -sin(phi)*cos(theta)]';
            J = [ eye(3) (1/rho)*dm_dtheta (1/rho)*dm_dphi -mi/(rho^2) ];
            J_all = sparse(double([eye(initialPositionOfFeature-1) zeros(initialPositionOfFeature-1,6) zeros(initialPositionOfFeature - 1, size_X_old - initialPositionOfFeature - 5);...
                zeros(3, initialPositionOfFeature-1) J zeros(3, size_X_old - initialPositionOfFeature - 5);...
                zeros(size_X_old - initialPositionOfFeature - 5, initialPositionOfFeature - 1) zeros(size_X_old - initialPositionOfFeature - 5, 6) eye(size_X_old - initialPositionOfFeature - 5)]));
            P = J_all*P*J_all';
            filter = set_x_k_k(filter,X);
            filter = set_p_k_k(filter,P);
            features_info(i).type = 'cartesian';
            return; % for the moment, only convert one feature per step (sorry!)
        end
    end
end