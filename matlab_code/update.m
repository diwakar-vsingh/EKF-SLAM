function [ x_k_k, p_k_k, K ] = update( x_km1_k, p_km1_k, H, R, z, h )

x_km1_k = sparse(double((x_km1_k)));
p_km1_k = sparse(double((p_km1_k)));
if size(z,1)>0

    % filter gain
    S = full(H*p_km1_k*H' + R);
    K = p_km1_k*H'*inv(S);

    % updated state and covariance
    x_k_k = full(x_km1_k) + K*( z - h );
    p_k_k = p_km1_k - K*S*K';
    p_k_k = 0.5*p_k_k + 0.5*p_k_k';
    % p_k_k = ( speye(size(p_km1_k,1)) - K*H )*p_km1_k;

    % normalize the quaternion
    Jnorm = normJac( x_k_k( 4:7 ) );
    size_p_k_k = size(p_k_k,1);
    p_k_k = [   p_k_k(1:3,1:3)              p_k_k(1:3,4:7)*Jnorm'               p_k_k(1:3,8:size_p_k_k);
        Jnorm*p_k_k(4:7,1:3)        Jnorm*p_k_k(4:7,4:7)*Jnorm'         Jnorm*p_k_k(4:7,8:size_p_k_k);
        p_k_k(8:size_p_k_k,1:3)     p_k_k(8:size_p_k_k,4:7)*Jnorm'      p_k_k(8:size_p_k_k,8:size_p_k_k)];

    x_k_k( 4:7 ) = x_k_k( 4:7 ) / norm( x_k_k( 4:7 ) );

else

    x_k_k = x_km1_k;
    p_k_k = p_km1_k;
    K = 0;

end