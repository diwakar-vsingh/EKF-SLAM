function f = ekf_update( f, features_info, cam )

[ f.x_k_k, f.p_k_k ] = update( f.x_k_km1, f.p_k_km1, f.H_matching,...
    f.R_matching, f.z, f.h, features_info, cam, f.measurements );