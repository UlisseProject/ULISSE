%% Test temperature
clear all; close all; clc; 

%% dati iniziali generali
I_sc_ref = 8.48; % [A]
V_oc_ref = 37.10; %[V]

Ns = 11 ; %numero di celle in serie, PER ORA A CASO PERCHE' SCRIPT DI JOEY CHE HO E' PRIMA DELLA SUA MODIFICA SULLE DIMENSIONI

T_V_OC=-0.29/100;% il nostro beta_VOC
T_I_SC=0.05/100;% il nostro alpha_ISC

%% Confronto con stessa Tamb e Gtot diverse
T_amb = 20;
Gtot1 = 800; 
Gtot2 = 900; 
Gtot3 = 1000; 

[I1, V1, P1, Tc1] = IVP(T_amb, Gtot1, I_sc_ref, V_oc_ref, Ns, T_V_OC, T_I_SC);
[I2, V2, P2, Tc2] = IVP(T_amb, Gtot2, I_sc_ref, V_oc_ref, Ns, T_V_OC, T_I_SC);
[I3, V3, P3, Tc3] = IVP(T_amb, Gtot3, I_sc_ref, V_oc_ref, Ns, T_V_OC, T_I_SC);

figure()
plot(V1, I1, V2, I2, V3, I3); 
str1 = sprintf('G = %d',Gtot1);
str2 = sprintf('G = %d',Gtot2);
str3 = sprintf('G = %d',Gtot3);
legend(str1, str2, str3);
xlabel('Voltage [V]');
ylabel('Current [A]');
title('I-V at different Tc with same Tc and different G');


figure()
plot(V1, P1, V2, P2, V3, P3); 
legend(str1, str2, str3);
xlabel('Voltage [V]');
ylabel('Power [W]');
title('P-V at different Tc');

%% Confronto con stessa T_amb e Gtot diverse
Gtot = 1000; 
T_amb1 = 0; 
T_amb2 = 15; 
T_amb3 = 30; 

[I1, V1, P1, Tc1] = IVP(T_amb1, Gtot, I_sc_ref, V_oc_ref, Ns, T_V_OC, T_I_SC);
[I2, V2, P2, Tc2] = IVP(T_amb2, Gtot, I_sc_ref, V_oc_ref, Ns, T_V_OC, T_I_SC);
[I3, V3, P3, Tc3] = IVP(T_amb3, Gtot, I_sc_ref, V_oc_ref, Ns, T_V_OC, T_I_SC);

figure()
plot(V1, I1, V2, I2, V3, I3); 
str1 = sprintf('Tc = %d',Tc1);
str2 = sprintf('Tc = %d',Tc2);
str3 = sprintf('Tc = %d',Tc3);
legend(str1, str2, str3);
xlabel('Voltage [V]');
ylabel('Current [A]');
title('I-V at different Tc');


figure()
plot(V1, P1, V2, P2, V3, P3); 
legend(str1, str2, str3);
xlabel('Voltage [V]');
ylabel('Power [W]');
title('P-V at different Tc with same G');








