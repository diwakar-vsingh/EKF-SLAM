function FRes = F_Test_dhu_dhrl( hrl, cam, hrl0 )

uv0 = hu( hrl0, cam);

nCols = size( hrl, 2 );
FRes = zeros( 2, nCols );

for j = 1:nCols
   FRes( :, j ) = hu( hrl( :, j ), cam ) - uv0;
end

return