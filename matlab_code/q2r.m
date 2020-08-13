function R=q2r(q)

x=q(2);
y=q(3);
z=q(4);
r=q(1);

R=[r*r+x*x-y*y-z*z  2*(x*y -r*z)     2*(z*x+r*y)
   2*(x*y+r*z)      r*r-x*x+y*y-z*z  2*(y*z-r*x)
   2*(z*x-r*y)      2*(y*z+r*x)      r*r-x*x-y*y+z*z];