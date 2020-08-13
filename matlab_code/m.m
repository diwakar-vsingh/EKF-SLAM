function m = m(a,b)
% Unit vector from azimut-elevation angles

if nargin<2
    
    theta = a(1,:); phi = a(2,:);
    cphi = cos(phi);
    m = [cphi.*sin(theta); -sin(phi); cphi.*cos(theta)];
    
else
    
    theta = a; phi = b;
    cphi = cos(phi);
    m = [cphi.*sin(theta)   -sin(phi)  cphi.*cos(theta)]';
    
end