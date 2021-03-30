function [I, V, P, Tc] = IVP(T_amb, Gtot, I_sc_ref, V_oc_ref, Ns, T_V_OC, T_I_SC)

NOCT = 50; % Temperatura in gradi, chiedere se va bene o è corretta 
Tc_ref = 25; % Temperatura in gradi di riferimento; 
Gref = 1000; % irradianza di riferimento
Gnoct = 800; 
Tnoct = 20; 
Tc = T_amb + (NOCT - Tnoct)*Gtot/Gnoct; %formula NOCT per temperatura della cella


n = 1.0134; % ideality factor (diode factor)


k = 8.6173324e-5; % ev/K 
Vt = n*k*(T_amb+273.15)*Ns;  % thermal voltage 


I_pv_ref = I_sc_ref; 
Vt_ref = n*k*(273.15)*Ns;
I0_ref = I_pv_ref/(exp(V_oc_ref/(Ns*Vt_ref)) - 1);

% Eg = 1.17 - 4.73*(10^-4)*(Tc+273.15).^2./(Tc + 273.15 + 636); 
% 
% Eg_ref = 1.17 - 4.73*(10^-4)*(Tc_ref + 273.15).^2./(Tc_ref + 273.15 + 636); 

I_pv = (Gtot/Gref)*I_pv_ref*(1 + T_I_SC*(Tc - Tc_ref));
% I0 = I0_ref*((Tc + 273.15)/(Tc_ref + 273.15))^3*exp((Eg_ref/(n*k*(Tc_ref + 273.15)))-(Eg/(n*k*(Tc + 273.15))));
V_oc = V_oc_ref*(1 + T_V_OC*(Tc - Tc_ref)) + (Ns*Vt*n)*log(Gtot/Gref);

I0 = I_pv/(exp(V_oc/(Ns*Vt)) - 1);

I_func = @(V) I_pv - I0*(exp(V/(Ns*Vt)) - 1);
V = linspace(0,V_oc,1000); 
I = I_func(V);
P = I.*V;
end
