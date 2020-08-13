function uv_c2=rotate_with_dist_fc_c2c1(cam, uv_c1, R_c2c1, t_c2c1, n, d)

% trasfer the image of points throug a camera rotation and translation
%   the camera has radial distortion
% Input 
%   cam    - camera calibration
%   uv_ini - initial camera postion point
%   R_c2c1 - camera rotation matrix
% Output
%  uv_rot - points on the rotated image

K = cam.K;

uv_c1_und=undistort_fm(uv_c1',cam)';
uv_c2_und=inv(K*(R_c2c1-(t_c2c1*n'/d))*inv(K))*[uv_c1_und';ones(1,size(uv_c1_und,1))];
uv_c2_und=(uv_c2_und(1:2,:)./[uv_c2_und(3,:);uv_c2_und(3,:)])';
uv_c2=distort_fm(uv_c2_und',cam)';