function FRes = F_Test_dhd_dhu( uv, cam, uv0 )

uv0 = distort_fm( uv0, cam );

nCols = size( uv, 2 );
FRes = zeros( 2, nCols );

for j = 1:nCols
   FRes( :, j ) = (distort_fm( uv( :, j ), cam ) - uv0);
end

return