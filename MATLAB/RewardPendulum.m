function [reward,k_end]=RewardPendulum(states,k, u_active, u_old)

x= states(1);
theta= states(2);
%x_dot= [0 0 1 0]*states;
theta_dot= states(4);
reward = 0;

if cos(theta) >= cos((177 * pi) / 180)
    k = 1;
    reward = reward - 0.1 * abs(u_active - u_old);
elseif cos(theta) < cos((177 * pi) / 180)
    k = min(k + 1, 40);  % Increment k but limit to 40
    reward = reward + 0.5 *k;
    reward = reward - 0.025* k * abs(u_active - u_old);
end

    % Determine the scaling factor
a = (k == 40) * 10 + (k < 40) * 0.5;

reward = reward+ 0.5 * (1-cos(theta)) -a*...
    (x/0.5)^2 -0.0003* theta_dot^2;
if cos(theta)<0
    reward = reward - 0.5 * cos(theta);
end
k_end = k;
end
