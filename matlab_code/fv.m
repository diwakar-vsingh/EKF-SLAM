function X_k_km1=fv(X_k_k,delta_t, type, std_a, std_alpha)

rW =X_k_k(1:3,1);
qWR=X_k_k(4:7,1);
vW =X_k_k(8:10,1);
wW =X_k_k(11:13,1);

if strcmp(type,'constant_orientation')
    wW = [0 0 0]';
    X_k_km1=[rW+vW*delta_t;
        qWR;
        vW;
        wW];
end

if strcmp(type,'constant_position')
    vW = [0 0 0]';
    X_k_km1=[rW;
        reshape(qprod(qWR,v2q(wW*delta_t)),4,1);
        vW;
        wW];
end

if strcmp(type,'constant_position_and_orientation')
    vW = [0 0 0]';
    wW = [0 0 0]';
    X_k_km1=[rW;
        qWR;
        vW;
        wW];
end

if strcmp(type,'constant_position_and_orientation_location_noise')
    vW = [0 0 0]';
    wW = [0 0 0]';
    X_k_km1=[rW;
        qWR;
        vW;
        wW];
end

if strcmp(type,'constant_velocity')
    X_k_km1=[rW+vW*delta_t;
        reshape(qprod(qWR,v2q(wW*delta_t)),4,1);
        vW;
        wW];
end