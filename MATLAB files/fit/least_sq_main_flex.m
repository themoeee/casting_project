global optim_data
%
scale_e=1;%1;
scale_s=600;
nb_total=7; % number of parameters = nb_total+2, nb_total is number of bezier points
params2optim = 5; % first 2 points have 3 params (sy, s1, d12), following only 1 (d_ii+1), last 2 (d_n-1n, sn)
%
% read/create reference
exp_file = fullfile("D:\Torsion_Test\virtual_experiments\SS2205_noiseless\input/UT_data.txt");
data = readmatrix(exp_file);%,"Sheet",3);
eps = data(:,1);
sig_ref = data(:,2);
%%
% overwrite reference for SV fitting
% eps_SV = linspace(0,0.223,1000)';
sv_params = define_sw_params();
% sig_ref_SV = compute_sw_point(sv_params,eps_SV);
% % merge data
% appendix = eps_SV>max(eps);
% eps = eps_SV;%[eps;eps_SV(appendix)];
% sig_ref = sig_ref_SV;%[sig_ref; sig_ref_SV(appendix)];
%%
[init_params,lb_vector,ub_vector]=define_init(nb_total);
lb_vector(params2optim+1:end) = 0.75;
ub_vector(params2optim+1:end) = 0.75;
%
objective=@(x) ls_objective_bezier_from_points(x,scale_e,scale_s,eps,sig_ref);
%
optim_data.x_all=[];
optim_data.f_all=[];
optim_data.g_all=[];
% optim_data.ref_params = sv_params;
optim_data.scale_e = scale_e;
optim_data.scale_s = scale_s;
options = optimoptions('fmincon','Algorithm','interior-point',...
    'MaxIterations',1000,'Display','none','MaxFunctionEvaluations',10^4,...
    'OutputFcn',@bezier_least_sq_export_iter);
[parameters_fit,objective_value]=fmincon(objective,init_params,[],[],[],[],lb_vector,ub_vector,[],options);
%
[cp_main,cp_secondary] = define_whip_bezier_control_points([parameters_fit(1);2*parameters_fit(2:end)-1],scale_e,scale_s);
yc = bezier_from_points(parameters_fit,scale_e,scale_s,eps);
figure
hold on
plot(eps,sig_ref,"-+")
plot(yc(:,1),yc(:,2))
plot(cp_main(:,1),cp_main(:,2),"o",cp_secondary(:,1),cp_secondary(:,2),"x")
xlabel("\epsilon")
ylabel("\sigma"),
legend("reference","fit","Location","southeast")
%%
% [~,yc] = objective(parameters_fit);
% figure
% hold on
% plot(eps,sig_ref)
% plot(yc(:,1),yc(:,2))
% legend("SV","WB","Location","southeast")
% xlabel("eps")
% ylabel("\sigma_y in MPa")
% title("nb = "+nb)
% 
eps_ext = linspace(0,2,1000)';
sig_ref_ext = compute_sw_point(sv_params,eps_ext);
[~,yc_ext] = ls_objective_bezier_from_points(parameters_fit,scale_e,scale_s,eps_ext,sig_ref_ext);
figure
hold on
plot(eps_ext,sig_ref_ext)
plot(yc_ext(:,1),yc_ext(:,2))
legend("SV","WB","Location","southeast")
xlabel("eps")
ylabel("\sigma_y in MPa")
title("scale_e = "+scale_e)