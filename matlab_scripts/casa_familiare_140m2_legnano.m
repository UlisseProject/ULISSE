clc
clear all

%legnano (mi)
Location.latitude = 45.5947;
Location.longitude = 8.91355;
Location.altitude = 199;

%location= [1:centro-nord 2:centro-sud 3:nord 4:sardegna 5:sicilia 6:sud]
location=3;

%contratto consumi
   
    %tariffa unica=1; tariffa per fasce=0-->numero fasce;
    tariffa=3;
    
    %mensilità in boletta
    lenght_periodo=2;
  
    %mesi inizio peridi
    mese_start=12;
    mesi(1)=mese_start;
    
    for i=2:12
        mesi(i)=mese_start+i-1;
        if mesi(i)>12
            mesi(i)=mesi(i)-12;
        end
    end
    
   
    consumi_import=[176 178 206 187 187 165 169 149 175 225 239 294 159 187 253 198 179 193];
    
    consumi_lenght=(12/lenght_periodo);
    for i=1:(consumi_lenght)
        t=(i-1)*3+1;
        consumi(i,1)=consumi_import(t);
        consumi(i, 2)=consumi_import(t+1);
        consumi(i, 3)=consumi_import(t+2);
    end
    
    %costo tariffa unica
    %costo=[0.1094 0.1094 0.1094 0.1093 0.1093 0.1093 0.1092 0.1092 0.1092 0.1092 0.1092 0.1092 0.1094 0.1094 0.1094 0.1092 0.1092 0.1092];
    
    
    %caso con 3 fasce di prezzo differenti
    costo=[0.1358 0.1144 0.1144 0.1366 0.1145 0.1145 0.1363 0.1144 0.1144  0.1364 0.1145 0.1145 0.1360 0.1137 0.1137 0.1357 0.1139 0.1139];
    costo(:,2)=costo(:,2)*1.1;
    
        consumi_lenght=(12/lenght_periodo);
    for i=1:(consumi_lenght)
        t=(i-1)*tariffa+1;
        for l=1:tariffa
        costi(i,l)=costo(t+l-1);
        end
    end
    
    %contratto pv system
        pv=0;
       %% 
        irr_mensile_pv= [590.38 591.36 798.42 1138.62 1613.50 2464.42 2934.12 3155.18 2759.40 1911.70 739.90 653.24];
        eff=0.1;
        prod_mensile_pv=irr_mensile_pv'*eff;
        

       