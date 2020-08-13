function FRes = F_Test_dh_dy( y, cam, Xv_km1_k, y0 )

h0 = hi( y0, Xv_km1_k, cam);

nCols = size( y, 2 );
FRes = zeros( 2, nCols );

for j = 1:nCols
   FRes( :, j ) = hi( y( :, j ), Xv_km1_k, cam ) - h0;
end

return