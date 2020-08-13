function uv_u = hu( yi, cam)

u0 = cam.Cx;
v0 = cam.Cy;
f  = cam.f;
ku = 1/cam.dx;
kv = 1/cam.dy;

uv_u = zeros( 2, size( yi, 2 ) );

for i = 1:size( yi, 2 )
    uv_u( 1, i ) = u0 + (yi(1,i)/yi(3,i))*f*ku;
    uv_u( 2, i ) = v0 + (yi(2,i)/yi(3,i))*f*kv;
end