function FRes = F_test_q2rpy( q, q0 )

nCols=size(q,1);
FRes=zeros(3,nCols);
for j=1:nCols
   FRes(:,j)=(tr2rpy(q2tr(q))-(tr2rpy(q2tr(q0))))';
end
return