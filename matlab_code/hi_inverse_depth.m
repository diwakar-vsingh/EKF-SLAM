function zi = hi_inverse_depth( yinit, t_wc, r_wc, cam, features_info )

% Compute a single measurement
% Javier Civera, 17/11/05

% Points 3D in camera coordinates
r_cw = r_wc';

yi = yinit(1:3);
theta = yinit(4);
phi = yinit(5);
rho = yinit(6);

mi = m( theta,phi );

hrl = r_cw*( (yi - t_wc)*rho + mi );

% % Angle limit condition: 
% % If seen 45º from the first time it was seen, do not predict it
% v_corig_p = rho*(features_info.r_wc_when_initialized - t_wc) + mi;
% v_c_p = mi;
% alpha = acos(v_corig_p'*v_c_p/(norm(v_corig_p)*norm(v_c_p)));
% if abs(alpha)>pi/4
%     zi = [];
%     return;
% end
% 
% % Scale limit condition:
% % If seen from double or half the scale, do not predict it
% scale = norm(v_corig_p)/norm(v_c_p);
% if (scale>2)||(scale<1/2)
%     zi = [];
%     return;
% end

% Is in front of the camera?
if ((atan2( hrl( 1, : ), hrl( 3, : ) )*180/pi < -60) ||...
    (atan2( hrl( 1, : ), hrl( 3, : ) )*180/pi > 60) ||...
    (atan2( hrl( 2, : ), hrl( 3, : ) )*180/pi < -60) ||...
    (atan2( hrl( 2, : ), hrl( 3, : ) )*180/pi > 60))
    zi = [];
    return;
end

% Image coordinates
uv_u = hu( hrl, cam );
% Add distortion
uv_d = distort_fm( uv_u , cam );

% Is visible in the image?
if ( uv_d(1)>0 ) && ( uv_d(1)<cam.nCols ) && ( uv_d(2)>0 ) && ( uv_d(2)<cam.nRows )
    zi = uv_d;
    return;
else
    zi = [];
    return;
end