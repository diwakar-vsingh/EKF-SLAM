function FRes = F_Test_dg_dxv( Xv, uvd, camera, lambdaInit, Xv_0 )

h0 = hinv( uvd, Xv_0, camera, lambdaInit );

nCols = size( Xv, 2 );
FRes = zeros( 6, nCols );

for j = 1:nCols
   FRes( :, j ) = hinv( uvd, Xv, camera, lambdaInit ) - h0;
end

return