function [points_curve] = compute_bezier_point(points_polygon,t_range)
%
n=size(points_polygon,1);
n1=n-1;
%
for i=0:1:n1
    sigma(i+1)=factorial(n1)/(factorial(i)*factorial(n1-i));  % for calculating (x!/(y!(x-y)!)) values
end
points_temp=[];
UB=[];
for t=t_range
    for d=1:n
        UB(d)=sigma(d)*((1-t)^(n-d))*(t^(d-1));
    end
    points_temp=cat(1,points_temp,UB);
end
points_curve=points_temp*points_polygon;
end

