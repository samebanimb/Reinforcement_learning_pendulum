
function [epsilon, alpha,x_dot,alpha_dot]= DynamicPendulum(X, u)
dt= 0.05;
m = 0.127;
M = 0.94;
l = 0.2066;
g =9.03788858603972;
Am= 8.28866833710732;
Beq = 1.16342493738728;
f = @(x,u) [x(3);
            x(4);
           (m*l*sin(x(2))*x(4)^2+m*g*sin(x(2))*cos(x(2))+(-Am*x(3)+Beq*u))/(M + m*sin(x(2))^2);
            -(m*l*sin(x(2))*cos(x(2))*x(4)^2+(m+M)*g*sin(x(2))+cos(x(2))*(-Am*x(3)+Beq*u))/(l*(M + m*sin(x(2))^2))]; % dx/dt = f(x,u)

%Km =0.007677634454753;
%Kt = 0.007682969729280;
%Kg=3.710000000000000;
%Mp =0.127000000000000;
%eta_g = 1;
%lp = 0.177800000000000;
%Jp = 0.001198730801458;
%r_mp = 0.006350000000000;
%Rm = 2.600000000000000;
%Bp = 0.002400000000000;
%Jeq= 1.073129910054978;
%Beq = 5.400000000000000;
%g = 9.810000000000000;

%f = @(x,u) [x(3);
%            x(4);
%            (- Km*Kt*x(3)*Kg^2*Mp*eta_g*lp^2 - Jp*Km*Kt*x(3)*Kg^2*eta_g +...
%            Kt*u*Kg*Mp*eta_g^2*lp^2*r_mp + Jp*Kt*u*Kg*eta_g^2*r_mp +...
%            Rm*sin(x(2))*Mp^2*x(4)^2*lp^3*r_mp^2 + Rm*g*cos(x(2))*sin(x(2))*Mp^2*lp^2*r_mp^2 +...
 %           Jp*Rm*sin(x(2))*Mp*x(4)^2*lp*r_mp^2 + Bp*Rm*cos(x(2))*Mp*x(4)*lp*r_mp^2 -...
 %           Beq*Rm*x(3)*Mp*lp^2*r_mp^2 - Beq*Jp*Rm*x(3)*r_mp^2)/(Rm*r_mp^2*(-...
%            Mp^2*lp^2*cos(x(2))^2 + Mp^2*lp^2 + Jeq*Mp*lp^2 + Jp*Mp + Jeq*Jp));
%            
%        -(Bp*Jeq*Rm*x(4)*r_mp^2 + Bp*Mp*Rm*x(4)*r_mp^2 + Mp^2*Rm*g*lp*r_mp^2*sin(x(2)) -...
%        Beq*Mp*Rm*lp*r_mp^2*x(3)*cos(x(2)) + Jeq*Mp*Rm*g*lp*r_mp^2*sin(x(2)) +...
%        Mp^2*Rm*x(4)^2*lp^2*r_mp^2*cos(x(2))*sin(x(2)) - Kg^2*Km*Kt*Mp*eta_g*lp*x(3)*cos(x(2)) +...
%        Kg*Kt*Mp*eta_g^2*lp*r_mp*u*cos(x(2)))/(Rm*r_mp^2*(- Mp^2*lp^2*cos(x(2))^2 + Mp^2*lp^2 + Jeq*Mp*lp^2 + Jp*Mp + Jeq*Jp))
%            ];
k1 = f(X, u);
k2 = f(X+dt/2*k1, u);
k3 = f(X+dt/2*k2, u);
k4 = f(X+dt*k3, u);
x_next = X+ dt/6*(k1+2*k2+2*k3+k4);
epsilon= x_next(1);
alpha= x_next(2);
x_dot=  x_next(3);
alpha_dot= x_next(4);

end