%% script batteria
clear all; close all;clc; 
%%
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

% coefficienti = [0.9556    0.0009   -0.0123    0.0079    0.0042   -0.2064   -0.0115   -0.0045    0.1199];

% p00 + p10*x + p01*y + p20*x^2 + p11*x*y + p02*y^2 + p21*x^2*y 
%                    + p12*x*y^2 + p03*y^3

% [X,Y] = meshgrid(a,b); 
% eta = @(x,y) coeff(1) + coeff(2).*x + coeff(3)*y + coeff(4)*x.^2 + coeff(5)*x.*y + coeff(6)*y.^2 + coeff(7)*x.^2.*y + coeff(8)*x.*y.^2 + coeff(9)*y.^3;
% z = surf(X,Y,eta(X,Y));

%% prova regressione polinomiale
[m,n] = size(eta_BESS);
% y = reshape(eta_BESS,m*n,1); % riordinato per colonne  

X = [x1,x2,x1.^2, x2.^2, x1.*x2, x1.^2.*x2, x2.^2.*x1, x1.^3, x2.^3];
X = [ones(m*n,1), X];

[theta] = normalEqn(X, y);

eta = @(p,s) theta(1) + theta(2)*p +theta(3)*s +theta(4)*p.^2 +theta(5)*s.^2 +theta(6)*p.*s +theta(7)*p.^2.*s +theta(8)*p.*s.^2 + theta(9)*p.^3 +theta(10)*s.^3;


a = linspace(0,1,100);
b = linspace(0,100,100);

[A,B] = meshgrid(a,b);
z = surf(A,B,eta(A,B));

hold on
P = [P_axis',P_axis',P_axis',P_axis',P_axis'];
S = [SOC;SOC;SOC;SOC;SOC;SOC;SOC;SOC;SOC];
surf(P,S,eta_BESS)
hold off





