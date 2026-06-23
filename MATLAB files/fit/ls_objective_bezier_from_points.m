function [loss, points_pred] = ls_objective_bezier_from_points(parameters, scale_e, scale_s, eps, sig_ref)
%
%
% change scale from [0,1] to [-1,1] for bezier parameters(2:end) (first is
% scale for initial yield stress and is in [0,inf)
parameters(2:end) = 2*parameters(2:end) - 1;
%
nb=length(parameters)-2;
points_ref=eps;
nref=size(points_ref,1);
points_pred=[points_ref(:,1),zeros(nref,1)];
[cp_main,cp_secondary] = define_whip_bezier_control_points(parameters,scale_e,scale_s);
%
index_within_interval=zeros(nref,nb-1);
for i=1:nb
    if i<nb % no extrapolation
        cond_low=(points_ref(:,1) >= cp_main(i));
        cond_high=(points_ref(:,1) <= cp_main(i+1));
        index_within_interval(:,i) = cond_low.*cond_high; % logical and for the 2 conditions
    else % extrapolation
        cond_low=(points_ref(:,1) >= cp_main(i));
        index_within_interval(:,i) = cond_low; % logical and for the 2 conditions
    end
end

for i=1:nb
    if i<nb % interpolate
        X0 = cp_main(i,1); % extracted from break points
        Y0 = cp_main(i,2);
        X1 = cp_secondary(i,1); % extracted from control points
        Y1 = cp_secondary(i,2);
        X2 = cp_main(i+1,1); % extracted from break points
        Y2 = cp_main(i+1,2);
        for j=find(index_within_interval(:,i))'
            eqps=points_pred(j,1);
            A_QUAD = X0 + X2 - 2*X1;
            B_QUAD = 2*(X1 - X0);
            C_QUAD = X0 - eqps;
            %
            if A_QUAD ~= 0
                DISCR_QUAD = B_QUAD^2 - 4 * A_QUAD * C_QUAD;
                t = (-B_QUAD + sqrt(DISCR_QUAD)) / (2 * A_QUAD);
                % here you need some additional proofs. It can be shown that:
                %
                % DISCR_QUAD >= 0. Treat DISCR_QUAD as f(X1) and compute its
                % minimum, which is -(eqps - X0)*(eqps - X2) >= 0
                %
                % The solution with -sqrt(DISCR_QUAD) is always out of range.
                % For A_QUAD > 0 the critical condition is t >= 0
                % For A_QUAD < 0 the critical condition is t <= 1
                %
            else % no longer an actual quadratic
                t = -C_QUAD / B_QUAD;
            end
            %
            points_pred(j,2) = Y1 + (1 - t)^2 * (Y0 - Y1) + t^2 * (Y2 - Y1);
        end
    else % extrapolate linearly
        slope = (cp_main(end,2) - cp_secondary(end,2)) /...
                (cp_main(end,1) - cp_secondary(end,1));
        for j=find(index_within_interval(:,i))'
            points_pred(j,2) = cp_main(end,2) + slope * (points_ref(j,1) - cp_main(end,1));
        end
    end
end
%
loss = sum((points_pred(:,2)-sig_ref).^2);
end

