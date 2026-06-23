function [material_vector] = define_sw_params()
%
% DP590
% alpha=0.7;    
% A=1126.19553;
% e0=0.001966;
% n_hard=0.205778;
% k0=375.513665;
% Q=404.18082;
% beta=18.140704;
%
% SS2205
alpha=0.126;    
A=2815.98;
e0=1e-6;
n_hard=0.47245;
k0=844.09;
Q=253.6;
beta=107.002;
%
material_vector=[alpha;A;e0;n_hard;k0;Q;beta];
end