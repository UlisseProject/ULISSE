
theta_r=(sin(n_air/n_glass)*sin(theta));

tau= @(theta, theta_r) e^(-K*L/cos(theta_r))*(1-1/2*((sin(theta_r-theta))^2/(sin(theta_r+theta))^2+(tan(theta_r-theta))^2/(tan(theta_r-theta))^2));
tau_b=tau(theta, theta_r);

beta_grad=rad2deg(beta);
theta_equiv_diff_deg=59.7-0.1388*beta_deg+0.001497*beta_deg^2;
theta_equiv_diff=deg2rad(theta_equiv_diff_deg);

tau_d=tau(theta_equiv_diff, theta_r);

tau_0=e^(-K*L)*(1-((n_glass-n_air)/(n_glass+n_air))^2);
K_tau_b=tau_b/tau_0;
K_tau_d=tau_d/tau_0;


