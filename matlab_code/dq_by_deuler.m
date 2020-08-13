function d=dq_by_deuler(euler_angles)

phi = euler_angles(1);
theta = euler_angles(2);
psi = euler_angles(3);

d = [(0.5)*(-sin(phi/2)+cos(phi/2))     (0.5)*(-sin(theta/2)+cos(theta/2))      (0.5)*(-sin(psi/2)+cos(psi/2)); 
     (0.5)*(+cos(phi/2)+sin(phi/2))     (0.5)*(-sin(theta/2)-cos(theta/2))      (0.5)*(-sin(psi/2)-cos(psi/2)); 
     (0.5)*(-sin(phi/2)+cos(phi/2))     (0.5)*(+cos(theta/2)-sin(theta/2))      (0.5)*(-sin(psi/2)+cos(psi/2)); 
     (0.5)*(-sin(phi/2)-cos(phi/2))     (0.5)*(-sin(theta/2)-cos(theta/2))      (0.5)*(+cos(psi/2)+sin(psi/2))];