function [SOC, P_out, n_cicli, eta] = flusso_potenza(P_in, SOC, Dt, Vn, i, SOC_max, n_cicli, Pn)
% P_in > 0 in carica
% P_in < 0 se scarica
% SOC = SOC della batteria 
% Vn nominal voltage
% i  current
% P_out, diverso da zero solo se SOC scende 

% !!! Mettere numero di cicli !!!
SOC_prev = SOC;

E_n = Pn*Dt; % W*s
E_max = SOC_max*3600;
Cn = E_n/Vn;  %nominal capacity

if P_in > 0 %ciclo di carica
   
    E_ch = P_in*Dt; %Energia per caricare storage

%     SOC = SOC + 1/Cn*i*Dt*100;
    if(E_ch<E_n) % P_in < Pn 
        SOC = SOC + E_ch/E_max;
        P_out = 0; 
        if SOC > 1
            SOC1 = SOC; 
            SOC = 1;
            disp("Batteria completamente carica");
            % we re not using i Cn1 = i*Dt/(SOC1 - SOC)*100;
            E_ch1 = (SOC1 - SOC)*E_n ; 
            P_out = E_ch1/Dt;    
        end
    else 
        disp("Superata potenza massima storata");
    end
        n_cicli = n_cicli + (SOC - SOC_prev);
        eta = 1; 
else    
    
    eta = calc_efficiency(SOC, abs(P_in), n_cicli, Pn); % !!! ATTENZIONE: l iput di SOC deve essere in percentuale
    E_dis = 1/eta*P_in*Dt; %Energia per caricare storage
%    Cn = E_dis/Vn;  %nominal capacity
    if(E_dis<E_n) % 1/eta*P_in < Pn
        SOC = SOC + E_dis/E_max; 
        P_out = 0; 
        if SOC < 0
            SOC1 = SOC; 
            SOC = 0;
            disp("Batteria completamente scarica");
            % not using i anymore Cn1 = i*Dt/(SOC1)*100;
            E_ch1 = (SOC1 - SOC)/100*E_n ; 
            P_out = E_ch1/Dt;   
        end
    else 
        disp("Superata potenza massima storata");
    end
        
end
