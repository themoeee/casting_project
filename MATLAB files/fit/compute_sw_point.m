function [sy,slope] = compute_sw_point(material_params,eqps)
%
if length(material_params) == 7
    has_alpha=1;
elseif length(material_params) == 6
    has_alpha=0;
else
    disp('Incorrect size for material_params in compute_sw_point.')
    return
end
% Unpacking the material parameter vector / make sure the call respects
% this order
if has_alpha
    alpha=material_params(1);    
    A=material_params(2);
    e0=material_params(3);
    n_hard=material_params(4);
    k0=material_params(5);
    Q=material_params(6);
    beta=material_params(7);
else
    A_new=material_params(1); % =alpha*A
    e0=material_params(2);
    n_hard=material_params(3);
    k0_new=material_params(4); % =(1-alpha)*k0
    Q_new=material_params(5); % =(1-alpha)*Q
    beta=material_params(6);
end
%
if has_alpha
    sy=alpha*A*(eqps+e0).^n_hard  +  (1-alpha)*(k0+Q*(1-exp(-beta*eqps)));
    slope=n_hard*alpha*A*(eqps+e0).^(n_hard-1)  +  (1-alpha)*Q*beta*exp(-beta*eqps);
else
    sy=A_new*(eqps+e0).^n_hard  +  (k0_new+Q_new*(1-exp(-beta*eqps)));
    slope=n_hard*A_new*(eqps+e0).^(n_hard-1)  +  Q_new*beta*exp(-beta*eqps);
end