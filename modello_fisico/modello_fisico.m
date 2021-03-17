%% Ramo 1
clear all; close all; clc;

addpath functions
addpath 'PVLib 1.4 Release'
addpath 'PVLib 1.4 Release\Required Data'
%% Dati iniziali
L = 2.5e-3; %valore di default in metri dello spessore del vetro
K = 10; %valore in metri del fattore di estinzione del vetro

n_air = 1.0002926; 
n_glass = 1.7; %da cercare valore tipico pv 

beta = deg2rad(30); %tilt angle

Location.latitude = 45.4773;
Location.longitude =  9.1815;
Location.altitude = 150;

%% Prova pvl_spa

Albedo=0.25;
z=zeros(24*60, 365);
alpha=zeros(24*60,365);
GHI=zeros(24*60, 365);
DNI=zeros(24*60, 365);
DHI=zeros(24*60, 365);
GR =zeros(24*60, 365);
% Create 1-min time series for Jan 1, 2012
i=1;
for m=1:12
    for d=1:30
DN = datenum(2012, m,d):1/(24*60):datenum(2012, m, d, 23, 59, 59);
Time = pvl_maketimestruct(DN, 1);
[SunAz, SunEl, ApparentSunEl, SolarTime]=pvl_ephemeris(Time, Location);
[SunAz1, SunEl1, ApparentSunEl1]=pvl_spa(Time, Location);
[ClearSkyGHI, ClearSkyDNI, ClearSkyDHI]= pvl_clearsky_ineichen(Time, Location);
dHr = Time.hour+Time.minute./60+Time.second./3600; % Calculate decimal hours for plotting
z(:, i)=SunEl1;
alpha(:, i)=SunAz1;
GHI(:, i)=ClearSkyGHI;
DNI(:, i)=ClearSkyDNI;
DHI(:, i)=ClearSkyDHI;
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
%% Formule irradianza
% 
% tau= @(theta, theta_r) e^(-K*L/cos(theta_r))*(1-1/2*((sin(theta_r-theta))^2/(sin(theta_r+theta))^2+(tan(theta_r-theta))^2/(tan(theta_r-theta))^2));
% theta = acos(cos(z)*cos(beta) + sin(z)*sin(beta)*sin(alpha-gamma));
% theta_r=(sin(n_air/n_glass)*sin(theta));
% 
% Gtot = Gdni*cos(theta) + Gdiff* (1+cos(beta))/2 + Grefl*(1-cos(beta))/2; 
% 
% G = tau_b*Gdni*cos(theta) + tau_d*Gdiff*(1+cos(beta))/2; 

