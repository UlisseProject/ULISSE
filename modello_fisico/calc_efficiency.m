function [eta] = calc_efficiency(SOC1, P_in, n_cicli, Pn)

eta_BESS=[0.540 0.540 0.550 0.480 0.480;
          0.540 0.540 0.550 0.480 0.480;
          0.842 0.842 0.842 0.787 0.787;
          0.818 0.818 0.931 0.896 0.896;
          0.926 0.926 0.947 0.917 0.917;
          0.895 0.895 0.931 0.927 0.927;
          0.868 0.868 0.922 0.908 0.908;
          0.861 0.861 0.896 0.859 0.859;
          0.861 0.861 0.896 0.859 0.859];


      
P_axis=[0.00 0.05 0.09 0.18 0.36 0.54 0.72 0.9 1];
SOC=[0 15 50 85 100];

x1=P_axis;
% x=SOC;
x2 = [SOC(1)*ones(9,1);SOC(2)*ones(9,1);SOC(3)*ones(9,1);SOC(4)*ones(9,1);SOC(5)*ones(9,1)];


x1=[x1'; x1'; x1'; x1'; x1'];
% x=[x'; x'; x'; x'; x'; x'; x'; x'; x'];


y=eta_BESS;
y=[y(:,1); y(:, 2); y(:,3); y(:,4); y(:,5)];
[m,n] = size(eta_BESS);

% y = reshape(eta_BESS,m*n,1); % riordinato per colonne  

X = [x1,x2,x1.^2, x2.^2, x1.*x2, x1.^2.*x2, x2.^2.*x1, x1.^3, x2.^3];
X = [ones(m*n,1), X];

[theta] = normalEqn(X, y);

eta1 = @(p,s) theta(1) + theta(2)*p +theta(3)*s +theta(4)*p.^2 +theta(5)*s.^2 +theta(6)*p.*s +theta(7)*p.^2.*s +theta(8)*p.*s.^2 + theta(9)*p.^3 +theta(10)*s.^3;

coeff_cicli = [0.0000   -0.0002    0.9965];
eff_cicli= polyval(coeff_cicli, n_cicli);

eta = eta1(P_in/Pn, SOC1)*eff_cicli;


