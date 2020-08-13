function [scrc,crc] = crosscorr(I1,I2,SVD)

% Check input parameters
sz = ones(1,3);
sz(1:ndims(I1)) = size(I1);
if prod(double((size(I2)==size(I1))))==0 %jmmm because prod didn't multipy logical values
  error(' ');
end

% The same with singular values (invariant to rotations)
if (nargin==3)&&(SVD=='svd')
  [scrc,crc] = crosscorrsvd(I1,I2);
  return
end

% Compute the normalized cross-correlation
flag= 1; %See std for details
num = (I1-repmat(mean(mean(I1,1),2),sz(1:2))).*(I2-repmat(mean(mean(I2,1),2),sz(1:2)));
den = repmat(std(reshape(I1,[1,prod(sz(1:2)),sz(3)]),flag,2),sz(1:2))...
    .*repmat(std(reshape(I2,[1,prod(sz(1:2)),sz(3)]),flag,2),sz(1:2));
crc = (den~=0).*num./(den+(den==0));
scrc= reshape(mean(mean(crc,1),2),[1,size(crc,3)]);

return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Auxiliar functions

function [scrc,crc] = crosscorrsvd(I1,I2)

flag = 1;
crc = [];
for i=1:size(I1,3)
  d1 = svd(I1(:,:,i)); d1 = diag(d1);
  d2 = svd(I2(:,:,i)); d2 = diag(d2);
  num = (d1-mean(d1,1)).*(d2-mean(d2,1));
  den = repmat(std(d1,flag,1).*std(d2,flag,1),size(num));
  crc = cat(3,crc,(den~=0).*num./(den+(den==0)));
end
scrc = mean(crc,1);

return


