function q=v2q(v)

% V2Q(R) converts rotation vector to quaternion.
%
%     The resultant quaternion(s) 
%          v_n=v/norm(v);
%          theta=norm(v);


theta=norm(v);
if (theta <eps)
    q=[1 0 0 0];
else
    v_n=v/norm(v);
    q=quaternions(v_n,theta);
end