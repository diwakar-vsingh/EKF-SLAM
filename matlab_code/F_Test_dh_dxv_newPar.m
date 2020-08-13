function FRes = F_Test_dh_dxv_newPar( Xv_km1_k, yi, cam, Xv_km1_k0 )

h0 = hi_newPar( yi, Xv_km1_k0, cam);

nCols = size( Xv_km1_k, 2 );
FRes = zeros( 2, nCols );

for j = 1:nCols
   FRes( :, j ) = hi_newPar( yi, Xv_km1_k( :, j ), cam ) - h0;
end

return