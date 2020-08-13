function FRes = F_Test_dg_dhd( uvd, Xv, camera, lambdaInit, uvd_0 )

h0 = hinv( uvd_0, Xv, camera, lambdaInit );

nCols = size( uvd, 2 );
FRes = zeros( 6, nCols );

for j = 1:nCols
   FRes( :, j ) = hinv( uvd, Xv, camera, lambdaInit ) - h0;
end

return