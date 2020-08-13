function cam = initialize_cam()

d =     0.0112;
nRows = 240;
nCols = 320;
Cx =    1.7945 / d;
Cy =    1.4433 / d;
k1=     6.333e-2;
k2=     1.390e-2;
f =     2.1735;

cam.k1      = k1;
cam.k2      = k2;
cam.nRows   = nRows;
cam.nCols   = nCols;
cam.Cx      = Cx;
cam.Cy      = Cy;
cam.f       = f;
cam.dx      = d;
cam.dy      = d;
cam.model = 'two_distortion_parameters';

cam.K =     sparse([ f/d   0     Cx;
                      0   f/d    Cy;
                      0    0     1]);
end