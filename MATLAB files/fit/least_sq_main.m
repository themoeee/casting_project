global optim_data
%
scale_e=0.7;
scale_s=600;
nb=7;
%
sv_params = define_sw_params();

eps = linspace(0,1.2*scale_e,1000)';
sig_ref = compute_sw_point(sv_params,eps);
%
% read file
% exp_file = fullfile("DP590/_Experiments_Overview.xlsx");
% data = readmatrix(exp_file,"Sheet",3);
% eps = data(:,1);
% sig_ref = data(:,2);
%%
objective=@(x) ls_objective_bezier_from_points(x,scale_e,scale_s,eps,sig_ref);
%
[init_params,lb_vector,ub_vector]=define_init(nb);
%
optim_data.x_all=[];
optim_data.f_all=[];
optim_data.g_all=[];
optim_data.ref_params = sv_params;
optim_data.scale_e = scale_e;
optim_data.scale_s = scale_s;
options = optimoptions('fmincon','Algorithm','interior-point',...
'MaxIterations',10000,'Display','final','MaxFunctionEvaluations',10^5,...
'OutputFcn',@bezier_least_sq_export_iter);
[parameters_fit,objective_value]=fmincon(objective,init_params,[],[],[],[],lb_vector,ub_vector,[],options);
%%
[~,yc] = objective(parameters_fit);
figure
hold on
plot(eps,sig_ref)
plot(yc(:,1),yc(:,2))
legend("SV","WB")
xlabel("eps")
ylabel("\sigma_y in MPa")
title("nb = "+nb)

eps_ext = linspace(0,2,1000)';
sig_ref_ext = compute_sw_point(sv_params,eps_ext);
[~,yc_ext] = ls_objective_bezier_from_points(parameters_fit,scale_e,scale_s,eps_ext,sig_ref);
figure
hold on
plot(eps_ext,sig_ref_ext)
plot(yc_ext(:,1),yc_ext(:,2))
legend("SV","WB")
xlabel("eps")
ylabel("\sigma_y in MPa")
title("nb = "+nb)