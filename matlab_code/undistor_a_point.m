function uvu = undistor_a_point( uvd, camera )

Cx = camera.Cx;
Cy = camera.Cy;
k1 = camera.k1;
k2 = camera.k2;
dx = camera.dx;
dy = camera.dy;

ud = uvd(1);
vd = uvd(2);
rd = sqrt( ( dx*(ud-Cx) )^2 + (dy*(vd-Cy) )^2 );

uvu = [ Cx + ( ud - Cx )*( 1 + k1*rd^2 + k2*rd^4 ); Cy + ( vd - Cy )*( 1 + k1*rd^2 + k2*rd^4 ) ];