function [yield_curve] = yield_curve_bezier(bezier_params, scale_e, scale_s, points_per_segment, extrapolation_max, extrapolation_stepwidth)
% parameters:
% bezier_params = [sy/scale_s, s1, d_12, d_23,... d_(n-1)n, sn]
% d -> slope of "main" polygon, between for example points 1 and 2
% s -> local slope of bezier curve at "main" polygon point
% size of bezier_params: [1, 1, nb-1, 1]
% calculation for bezier part is done dimensionless and at the end
% multiplied by scale_s resp. scale_e
% bezier_params(2:end) in (-1,1)
% points_per_segment specifies the number of points on the yield curve per
% bezier segment (n-1 subsegments)
% extrapolation_max is the span in epsilon for the yield curve to which it
% should be extended if scale_e is too small to reach that epsilon in the
% bezier part only
% extrapolation_stepwidth is the stepwidth if extrapolation is needed

% number of bezier points
nb = length(bezier_params)-2;
% calculation of yield curve
% bezier part
np_pb=points_per_segment; % points per bezier segment (divides the segment into np_bp-1 subsegments)
points_se_curve_bezier=zeros((nb-1)*(np_pb-1)+1,2);
[cp_main,cp_secondary] = define_whip_bezier_control_points(bezier_params,scale_e,scale_s);
for i=1:nb-1
    cp_input=[cp_main(i,:);cp_secondary(i,:);cp_main(i+1,:)];
    if i==nb-1 % last curve has all points
        t_range=linspace(0,1,np_pb);
        points_se_curve_bezier((i-1)*(np_pb-1)+1:end,:)=compute_bezier_point(cp_input,t_range);
    else % non-last curve has 1 less point
        t_range=linspace(0, 1-1/np_pb, np_pb-1);
        points_se_curve_bezier((i-1)*(np_pb-1)+1:i*(np_pb-1),:)=compute_bezier_point(cp_input,t_range);
    end
end
% linear part
e_max=extrapolation_max;
step_e=extrapolation_stepwidth;
if scale_e<e_max
    points_linear_e=((points_se_curve_bezier(end,1)+step_e):step_e:e_max)';
    slope=(cp_main(end,2)-cp_secondary(end,2))/(cp_main(end,1)-cp_secondary(end,1));
    points_linear_s=points_se_curve_bezier(end,2)+...
    slope*(points_linear_e-points_se_curve_bezier(end,1));
    %
    points_se_curve_linear=[points_linear_e,points_linear_s];
else
    points_se_curve_linear=[];
end
% concatenate
yield_curve=[points_se_curve_bezier;points_se_curve_linear];
end