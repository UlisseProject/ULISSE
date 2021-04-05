%% importa volari

clc
clear all
load('milanotemperature.mat')

%%
m=1;
c=1;
t=1;
s=1;
b=1;
lun=length(milanotemperature);
mancanti_2=zeros(599,1);
for i=1:lun
    if milanotemperature(i)==-999
        mancanti_2(t,1 )=mancanti_2(t,1)+1;
        if milanotemperature(i-1)==-999
        elseif milanotemperature(i+1)==-999
          bug(b)=i;
          b=b+1;
        end
        if or(milanotemperature(i+1)==-999, milanotemperature(i-1)==-999)
        mancanti(m)=i;
        m=m+1;
        
        
        else
            milanotemperature(i)=(milanotemperature(i+1)+milanotemperature(i-1))/2;
            s=s+1;
        end
    else
        t=t+1;
        if  mancanti_2(t-1,1 )==0
            t=t-1;
        end
    end    
end

totale_mancanti_blocco=m-1;
totale_mancanti_singoli=s-1;
totali_capire=c-1;
%% risolvi problema mancanti


milanotemperature(28027:(28027+569))=(milanotemperature((28027-144*4):(28027+569-144*4))+milanotemperature((28027+144*4):(28027+569+144*4)))/2;

milanotemperature(43566:(43566+30))=(milanotemperature((43566-144):(43566+30-144))+milanotemperature((43566+144):(43566+30+144)))/2;

%% creo matrice 1440*366

Tamb_matrix=zeros(1440, 366);

giorni_mese=[31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

g=0;

for m=1:12
    for d=1:giorni_mese(m)
        g=g+1;
        for timestep=1:1440
            if ((g-1)*144+(floor(timestep/10)))==52704
                Tamb_matrix(1440, 366)=(Tamb_matrix(1, 1)+Tamb_matrix(1439, 366))/2;
            else
            Tamb_matrix(timestep, g)=milanotemperature((g-1)*144+(floor(timestep/10)+1),1);
            end
        end
    end
end







%%        
% Legenda
% -999 Valore mancante o invalido
% Legenda dei valori speciali
% Valore, Descrizione
% 777,calma
% 7777,calma
% 888,variabile
% 8888,variabile
% Nota: i gradi di Direzione del Vento sono riferiti al Nord