function [SOC, P_out] = flusso_potenza(P_in, SOC, Dt, Vn, i, SOC_max)
% P_in > 0 in carica
% P_in < 0 se scarica
% SOC = SOC della batteria 
% Vn nominal voltage
% i  current
% P_out, diverso da zero solo se SOC scende 

% !!! Mettere numero di cicli !!!

if P_in > 0 %ciclo di carica
    E_ch = P_in*Dt; %Energia per caricare storage
    Cn = E_ch/Vn;  %nominal capacity
    SOC = SOC + 1/Cn*i*Dt; 
    P_out = 0; 
    if SOC > SOC_max
        SOC1 = SOC; 
        SOC = SOC_max;
        disp("Batteria completamente carica");
        Cn1 = i*Dt/(SOC1 - SOC);
        E_ch1 = Cn1*Vn; 
        P_out = E_ch/Dt;    
    end
     
    
else    
    eta = FUNZIONE_ENRICO(SOC, P_in); % !!! ATTENZIONE: l'iput di SOC deve essere in percentuale
    E_dis = 1/eta*P_in*Dt; %Energia per caricare storage
    Cn = E_dis/Vn;  %nominal capacity
    SOC = SOC + 1/Cn*i*Dt; 
    P_out = 0; 
    if SOC < 0
        SOC1 = SOC; 
        SOC = 0;
        disp("Batteria completamente scarica");
        Cn1 = i*Dt/(SOC1 - SOC);
        E_ch1 = Cn1*Vn; 
        P_out = eta*E_dis/Dt;   
    end
    
end
