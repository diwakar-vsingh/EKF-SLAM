function [ cartesian ] = inversedepth2cartesian( inverse_depth )

rw = inverse_depth(1:3,:);
theta = inverse_depth(4,:);
phi = inverse_depth(5,:);
rho = inverse_depth(6,:);

cphi = cos(phi);
m = [cphi.*sin(theta);   -sin(phi);  cphi.*cos(theta)];   
cartesian(1,:) = rw(1) + (1./rho).*m(1,:);
cartesian(2,:) = rw(2) + (1./rho).*m(2,:);
cartesian(3,:) = rw(3) + (1./rho).*m(3,:);