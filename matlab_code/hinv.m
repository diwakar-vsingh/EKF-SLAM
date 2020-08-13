function newFeature = hinv( uvd, Xv, camera, initial_rho )

  fku =  camera.K(1,1);
  fkv =  camera.K(2,2);
  U0  =  camera.K(1,3);
  V0  =  camera.K(2,3);
  
  uv = undistort_fm( uvd, camera );
  u = uv(1);
  v = uv(2);
  
  r_W  = Xv(1:3);
  q_WR = Xv(4:7);
  
  h_LR_x = -(U0-u)/fku;
  h_LR_y = -(V0-v)/fkv;
  h_LR_z = 1;
  
  h_LR=[h_LR_x; h_LR_y; h_LR_z];
  
  n=q2r(q_WR)*h_LR;
  nx=n(1);
  ny=n(2);
  nz=n(3);
  
  newFeature = [ r_W; atan2(nx,nz); atan2(-ny,sqrt(nx*nx+nz*nz)); initial_rho ];
  
return