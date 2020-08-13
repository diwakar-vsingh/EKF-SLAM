function FRes = F_Test_dh_dy_newPar( yi, Xv_km1_k, cam, yi_0 )

h0 = hi_newPar( yi_0, Xv_km1_k, cam);

nCols = size( yi, 2 );
FRes = zeros( 2, nCols );

for j = 1:nCols
   FRes( :, j ) = hi_newPar( yi( :, j ), Xv_km1_k, cam ) - h0;
end

return