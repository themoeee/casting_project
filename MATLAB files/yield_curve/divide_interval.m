function [vals] = divide_interval(edge1,edge2,parameters)
% This implementation assumes parameters range from -1 to 1
% Fist value will be edge1 and last value will be edge2
np=length(parameters);
remaining_range=1;
vals_dimensionless=[0;zeros(np,1);1];
%
for i=1:np
    vals_dimensionless(i+1)=...
    vals_dimensionless(i) + remaining_range * (1+parameters(i))/2;
    remaining_range=remaining_range-(vals_dimensionless(i+1)-vals_dimensionless(i));
end
vals=edge1+(edge2-edge1)*vals_dimensionless;
end

