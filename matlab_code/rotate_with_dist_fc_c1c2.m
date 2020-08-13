function uv_c1=rotate_with_dist_fc_c1c2(cam, uv_c2, R_c1c2, t_c1c2, n, d)

% trasfer the image of points throug a camera rotation and translation
%   the camera has radial distortion
% Input 
%   cam    - camera calibration
%   uv_ini - initial camera postion point
%   R_c1c2 - camera rotation matrix
% Output
%  uv_rot - points on the rotated image

K = cam.K;

uv_c2_und=undistort_fm(uv_c2',cam)';
uv_c1_und=K*(R_c1c2-(t_c1c2*n'/d))*inv(K)*[uv_c2_und';ones(1,size(uv_c2_und,1))];
uv_c1_und=(uv_c1_und(1:2,:)./[uv_c1_und(3,:);uv_c1_und(3,:)])';
uv_c1=distort_fm(uv_c1_und',cam)';