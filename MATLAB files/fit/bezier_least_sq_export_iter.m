function stop = bezier_least_sq_export_iter(x,optimValues,state)
%
global optim_data
%
stop = false;
%
optim_data.x_all=[optim_data.x_all,x];
optim_data.f_all=[optim_data.f_all,optimValues.fval];
optim_data.g_all=[optim_data.g_all,optimValues.gradient];
%
end
