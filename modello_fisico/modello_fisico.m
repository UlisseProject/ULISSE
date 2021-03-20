%% Ramo 1
clear all; close all; clc;

addpath functions
addpath 'PVLib 1.4 Release'
addpath 'PVLib 1.4 Release\Required Data'
%% Dati iniziali
L = 2.5e-3; %valore di default in metri dello spessore del vetro
K = 10; %valore in metri del fattore di estinzione del vetro

n_air = 1.0002926; 
n_glass = 1.58992; % da paper

beta = deg2rad(30); %tilt angle
gamma = deg2rad(0);

Location.latitude = 45.4773;
Location.longitude =  9.1815;
Location.altitude = 150;
load('Tamb_matrix.mat')

%% Prova pvl_spa

Albedo=0.1;
z=zeros(24*60, 366);
alpha=zeros(24*60,366);
GHI=zeros(24*60, 366);
DNI=zeros(24*60, 366);
DHI=zeros(24*60, 366);
GR =zeros(24*60, 366);
% Create 1-min time series for Jan 1, 2012
giorni_mese=[31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
i=1;
for m= 6:6
    for d = 1:giorni_mese(m)
        DN = datenum(2012, m,d):1/(24*60):datenum(2012, m, d, 23, 59, 59);
        Time = pvl_maketimestruct(DN, 1);
        [SunAz, SunEl, ApparentSunEl, SolarTime]=pvl_ephemeris(Time, Location);
        [SunAz1, SunEl1, ApparentSunEl1]=pvl_spa(Time, Location);
        [ClearSkyGHI, ClearSkyDNI, ClearSkyDHI]= pvl_clearsky_ineichen(Time, Location);
        dHr = Time.hour+Time.minute./60+Time.second./3600; % Calculate decimal hours for plotting
%        z(:, i)=deg2rad(SunEl1);
        z(:, i)=deg2rad(90- SunEl1);
        %alpha(:, i)=deg2rad(SunAz1);
        alpha(:, i)=deg2rad(SunAz1-180);
        GHI(:, i)=ClearSkyGHI;
        DNI(:, i)=ClearSkyDNI;
        DHI(:, i)=ClearSkyDHI;

        GHI(isnan(GHI))=0;
        DNI(isnan(DNI))=0;
        DHI(isnan(DHI))=0;

        GR(:, i) = pvl_grounddiffuse(rad2deg(beta), GHI(:,i), Albedo);

        i=i+1;
    end
end


% figure
% plot(dHr,ApparentSunEl)
% hold all
% plot(dHr,ApparentSunEl1)
% legend('ephemeris','spa')
% title('Solar Elevation Angle on Jan 1, 2012 in Milan, NM','FontSize',14)
% xlabel('Hour of the Day (hr)')
% ylabel('Solar Elevation Angle (deg)')
% dif = ApparentSunEl1-ApparentSunEl;
%% Coefficienti correttivi
Gdiff = DHI;
Gdni = DNI; 
Grefl = GR; 
%% Formule irradianza

alt_sol=deg2rad(SunEl1);
theta=acos(cos(alt_sol).*cos(alpha-gamma).*sin(beta)+sin(alt_sol).*cos(beta));
% theta = acos(cos(z).*cos(beta) + sin(z).*sin(beta).*sin(alpha-gamma));

theta_r = asin(n_air/n_glass.*sin(theta));
 
Gtot = Gdni.*cos(theta) + Gdiff.*(1+cos(beta))/2 + Grefl*(1-cos(beta))/2; 

%%
% f1 = @(theta_r) exp((-K*L./cos(theta_r)));
% f2 = @(theta, theta_r) (1-1/2.*((sin(theta_r-theta)).^2./(sin(theta_r+theta)).^2+(tan(theta_r-theta)).^2./(tan(theta_r+theta)).^2)))
% II = find(G(:,1));
% figure
% plot(theta_r(II,1), f1(theta_r(II,1)));
% figure
% plot(theta(II,1),f2(theta(II,1),theta_r(II,1)))

%%

tau= @(theta, theta_r) exp((-K*L./cos(theta_r))).*(1-1/2.*(((sin(theta_r-theta)).^2./(sin(theta_r+theta)).^2+(tan(theta_r-theta)).^2./(tan(theta_r+theta)).^2)));

tau_b=tau(theta, theta_r);
beta_deg=rad2deg(beta);

theta_equiv_diff_deg= 59.7 - 0.1388*beta_deg + 0.001497*beta_deg^2;
%theta_equiv_diff_deg= 90 - 0.5788*beta_deg + 0.002693*beta_deg^2;

theta_equiv_diff=deg2rad(theta_equiv_diff_deg);
tau_d=tau(theta_equiv_diff, theta_r);
tau_0=exp((-K*L)).*(1-((n_glass - 1)./(n_glass + 1)).^2);

K_tau_b=tau_b/tau_0;
K_tau_d=tau_d/tau_0;

% tau_d = 1; 
G = tau_b.*Gdni.*cos(theta) + tau_d.*Gdiff.*(1+cos(beta))/2; 

dHr = Time.hour+Time.minute./60+Time.second./3600;

%% plot section

figure(1)
plot(dHr, G(:,1:31))
title('Irradiance')
xlabel('time [h]')
ylabel('G [W/m^2]')

figure(2)
find(G(:,1)); 
II = find(G(:,1));
plot(rad2deg(theta(II,1)), K_tau_b(II,1))
title('Incidence angle modifier for direct irradiance (K_{tau_b})')
xlabel('theta [°]')
ylabel('K_{tau_b}')


figure(3)
find(G(:,1)); 
II = find(G(:,1));
plot(rad2deg(theta_r(II,1)), K_tau_d(II,1))
title('Incidence angle modifier for diffuse irradiance (K_{tau_d})')
xlabel('theta_r [°]')
ylabel('K_{tau_d}')


figure(4)
find(G(:,1)); 
II = find(G(:,1));
plot(rad2deg(theta(II,1)), tau_b(II,1))
title('Glass trasmittance for direct irradiance (tau_b)')
xlabel('theta [°]')
ylabel('tau_b')

figure(5)
find(G(:,1)); 
II = find(G(:,1));
plot(rad2deg(theta_r(II,1)), tau_d(II,1))
title('Glass trasmittance for direct irradiance (tau_d)')
xlabel('theta_r [°]')
ylabel('tau_d')

%% plot movimento angoli
figure(6)
plot(dHr, rad2deg(theta(:,1)));
title('Theta')
xlabel('time [h]')
ylabel('theta[°]')
hold on
plot([8 8], [0 max(max(rad2deg(theta(:,1:31))))], 'r', 'linew', 2)
plot([17 17], [0 max(max(rad2deg(theta(:,1:31))))], 'r', 'linew', 2)
hold off

% figure(7)
% plot(dHr, rad2deg(z(:,1)));
% title('Solar zenit angle z')
% xlabel('time [h]')
% ylabel('z[°]')
% hold on
% plot([8 8], [min(min(rad2deg(z(:,1:31)))) max(max(rad2deg(z(:,1:31))))], 'r', 'linew', 2)
% plot([17 17], [min(min(rad2deg(z(:,1:31)))) max(max(rad2deg(z(:,1:31))))], 'r', 'linew', 2)
% hold off

figure(8)
plot(dHr, rad2deg(alpha(:,1)));
title('Solar azimuth angle alpha')
xlabel('time [h]')
ylabel('alpha[°]')
hold on
plot([8 8], [0 max(max(rad2deg(alpha(:,1:31))))], 'r', 'linew', 2)
plot([17 17], [0 max(max(rad2deg(alpha(:,1:31))))], 'r', 'linew', 2)
hold off

% figure(9)
% plot(rad2deg(theta(:,1)), cos(theta(:,1))); %questo grafico viene uguale a immagine trovata
% title('cos(theta) as a function of theta')
% xlabel('theta[°]')
% ylabel('cos(theta)')

% figure(10)
% plot(dHr, cos(theta(:,1))); %tuttavia andamento di cos(theta) col tempo è strano
% title('cos(theta) as a function of time')
% xlabel('time [h]')
% ylabel('cos(theta)')
% hold on
% plot([8 8], [0 1], 'r', 'linew', 2)
% plot([17 17], [0 1], 'r', 'linew', 2)
% hold off

figure()
find(G(:,1)); 
II = find(G(:,1));
plot(rad2deg(theta(II,1)), K_tau_b(II,1))

%%
% %% Formula NOCT: formula e parametri presi da Temperature Dependent Power Modeling of Photovoltaics 2013
% 
% T_amb = 25; % INPUT ESTERNO DEL SISTEMA
% 
% NOCT = 45; % Temperatura in gradi, chiedere se va bene o è corretta 
% Tc_ref = 25; % Temperatura in gradi di riferimento; 
% eta_r = 0.129; % efficienza a temperatura di riferimento 
% T_coeff = 0.0048; %coefficiente di temperatura
% Gref = 1000; 
% 
% Tc = T_amb + (NOCT - Tref)*Gtot/Gref;
% 
% eta = eta_r*(1 - T_coeff*(Tc - T_ref));
% 
% 
% %% 
% alphaISC = ; % temperature coefficient for short circuit
% iPV_ref = ; %dato da trovare
% Eg=1.17-4.73*10^(-4)*Tc^2/(Tc+636);
% n = 1.0134; % ideality factor (diode factor)
% Vt = ; %thermal voltage, da capire
% 
% iPV=Gtot/Gref*(iPV_ref*(1+alphaISC*(Tc-Tref)));
% i0=io_ref*(Tc/Tref)^3*exp(Eg_ref/(n*k*Tref)-Eg/(n*k*Tc));







