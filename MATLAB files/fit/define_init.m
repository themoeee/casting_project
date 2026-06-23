function [init_params,lb,ub] = define_init(nb)
%
minmax_limit=0.998;
init_params=[1;0.5*ones(nb+1,1)];
lb=[0;zeros(nb+1,1)];
ub=[10;minmax_limit*ones(nb+1,1)];
end