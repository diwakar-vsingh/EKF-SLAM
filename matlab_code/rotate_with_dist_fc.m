function uv_rot=rotate_with_dist_fc(cam, uv_ini, R_c2c1, t_c2c1, n, d)

% trasfer the image of points throug a camera rotation and translation
%   the camera has radial distortion
% Input 
%   cam    - camera calibration
%   uv_ini - initial camera postion point
%   R_c2c1 - camera rotation matrix
% Output
%  uv_rot - points on the rotated image

uv_ini_und=undistort_fm(uv_ini',cam)';
uv_rot_und=inv(cam.K*(R_c2c1-(t_c2c1*n'/d))*inv(cam.K))*[uv_ini_und';ones(1,size(uv_ini_und,1))];
uv_rot_und=(uv_rot_und(1:2,:)./[uv_rot_und(3,:);uv_rot_und(3,:)])';
uv_rot=distort_fm(uv_rot_und',cam)';