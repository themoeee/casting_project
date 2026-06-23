function [cp_main,cp_secondary] = define_whip_bezier_control_points(parameters,scale_e,scale_s)
%
nb=length(parameters)-2;
%
% parameters = [sy, s1, d_12, d_23,... d_(n-1)n, sn]
% d -> slope of "main" polygon, between for example points 1 and 2
% s -> local slope of bezier curve at "main" polygon point
% a -> epicenter angle of "main" polygon point
%
% size of parameters: [1, 1, nb-1, 1]
%
% All calculations are made in dimentionless units and the final results
% are multiplied by scale_e and scale_s at the end
%
cp_main=zeros(nb,2);
cp_secondary=zeros(nb-1,2);
% Variable name convention:
% a_ -> angle, in rad
% s_ -> slope, i.e. tan(a_)
% 
% _a -> pertaining to epicenter angle
% _d -> pertaining to main polygon slope
% _s -> pertaining to actual curve slope
s_d=zeros(nb-1,1); % corresponding to d slopes, see comment above
s_s=zeros(nb,1); % corresponding to s slopes, see comment above
s_a=zeros(nb,1);
a_d=zeros(nb-1,1);
a_s=zeros(nb,1);
a_a=zeros(nb,1); % coresponding to a angles, see comment above
a_a(end)=pi/2; % last point has this angle by definition
%
% Defining the main polygon
%
sy=parameters(1);
%
cp_main(1,1)=0;
cp_main(1,2)=sy; % the first point can be defined directly
%
a_ds_init=divide_interval(pi/2,0,-parameters(2:nb+2));
% If parameters is used, -1 will correspond to high. To prevent this
% -parameters is used, so that 1 corresponds to high.
%
a_d=a_ds_init(3:end-2);
s_d=tan(a_d);
% skipping two on each side. (These are pi/2, s1 and sn, 0 respectively)
a_s(1)=a_ds_init(2);
a_s(nb)=a_ds_init(end-1);
%
% remaining secondary slopes
for i=2:nb-1
    %
    a_s_min=a_d(i);
    a_s_max=a_d(i-1);
    w_s=0; % all slopes are given values in the middle
    a_s(i)=a_s_min + (a_s_max-a_s_min) * (0.5*w_s+0.5);
    %
end
s_s=tan(a_s);
%
% angles (a) inversely proportional to s changes
for i=1:nb-2
    nr=nb-i;
    %
    % the slope changes to derive the following equation need to be
    % predicted, since computing the directly from the available slopes
    % leads to information leaking backwards. This prediction is made by
    % assuming that all d angles for the remaineder of the curve are
    % distributed equidistantly and subsequently placing the s angles in
    % the middle. This means that for most of the remaining d segments the
    % assumed change of slope is constant, which allows the use of the
    % following equations
    delta_a_s_i=a_s(i)-a_d(i)*(2*nr-1)/(2*nr);
    a_a(i+1)=a_a(i)+(pi/2-a_a(i))*(1/delta_a_s_i)/(1/delta_a_s_i+(nr-1)*nr/a_d(i));
    % Note that the index i for a angles is different than what is used in
    % the paper, although it might actually make sense for me to fix that.
end
s_a=tan(a_a);
%
for i=1:nb-1
    %
    % solve for intersection of line defined by previous point + slope with
    % the corresponding line defined by the "star" scheme
    if i==nb-1 % last point, vertical slope
        cp_main(i+1,1)=1;
        cp_main(i+1,2)=cp_main(i,2)+s_d(i)*(1-cp_main(i,1));
    else % other points, regural slope
        cp_main(i+1,1)=(sy+s_d(i)*cp_main(i,1)-cp_main(i,2)+s_a(i+1)) /...
                       (s_d(i)+s_a(i+1));
        cp_main(i+1,2)=cp_main(i,2)+s_d(i)*(cp_main(i+1,1)-cp_main(i,1));
    end
end
%
% solving for the intersection of consecutive slope lines
%
for i=1:nb-1
    cp_secondary(i,:)=quadratic_bezier_solve_for_middle_point(cp_main(i:i+1,:),s_s(i:i+1));
end
%
cp_main(:,1)=cp_main(:,1)*scale_e;
cp_main(:,2)=cp_main(:,2)*scale_s;
cp_secondary(:,1)=cp_secondary(:,1)*scale_e;
cp_secondary(:,2)=cp_secondary(:,2)*scale_s;
%
end

