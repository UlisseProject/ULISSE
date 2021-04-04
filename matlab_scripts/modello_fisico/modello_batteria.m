%% modello dello storage
clear all; close all; clc; 

%% coefficienti cicli

Pn = 250000; %W it s actually Pmax
SOC = 0; 
Dt = 1e12;% s
Vn = 700; %V 

V = 220; 
E_max = 571900; % [Wh] NOTE:way too big
i = 7; %A
P_in = -V*i; 
n_cicli = 0;

[SOC, P_out, n_cicli, eta] = flusso_potenza(P_in, SOC, Dt, Vn, i, E_max, n_cicli, Pn);
