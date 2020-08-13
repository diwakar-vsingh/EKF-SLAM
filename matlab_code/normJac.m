function J=normJac(q)

     r=q(1);
     x=q(2);
     y=q(3);
     z=q(4);

     J=(r*r+x*x+y*y+z*z)^(-3/2)*...
   [x*x+y*y+z*z         -r*x         -r*y         -r*z;
	       -x*r  r*r+y*y+z*z         -x*y         -x*z;
	       -y*r         -y*x  r*r+x*x+z*z         -y*z;
	       -z*r         -z*x         -z*y  r*r+x*x+y*y];

 return

