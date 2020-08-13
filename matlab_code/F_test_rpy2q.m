function FRes = F_test_rpy2q( rpy, rpy0 )

nCols=size(rpy,2);
FRes=zeros(4,nCols);
for j=1:nCols
   FRes(:,j)=(tr2q(rpy2tr(rpy(1,j),rpy(2,j),rpy(3,j)))-tr2q(rpy2tr(rpy0(1),rpy0(2),rpy0(3))))';
end
return