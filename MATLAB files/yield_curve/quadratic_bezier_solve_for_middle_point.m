function [middle_point] = quadratic_bezier_solve_for_middle_point(coordinates,slopes)
%
% coordinates -> 2x2 ([x1,y1;x2,y2])
% slopes -> 2x1 ([s1,s2])
x1=coordinates(1,1);
x2=coordinates(2,1);
y1=coordinates(1,2);
y2=coordinates(2,2);
%
s1=slopes(1);
s2=slopes(2);
%
A=zeros(2,2);
b=zeros(2,1);
%
% choose most stable form of the equations
%
if abs(s1)>1
    A(1,:)=[1, -1/s1];
    b(1)=x1-y1/s1;
else
    A(1,:)=[s1, -1];
    b(1)=s1*x1-y1;
end
if abs(s2)>1
    A(2,:)=[1, -1/s2];
    b(2)=x2-y2/s2;
else
    A(2,:)=[s2, -1];
    b(2)=s2*x2-y2;
end
%
middle_point=(A\b)';
end

