function [nu, S] = extract_nu_and_S(f)

nu = f.z - f.h;
S = f.S_matching;