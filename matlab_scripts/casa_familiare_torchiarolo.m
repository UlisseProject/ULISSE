
%location= [1:centro-nord 2:centro-sud 3:nord 4:sardegna 5:sicilia 6:sud]
location=6;
%torchairolo (br)

Location.latitude = 40.485;
Location.longitude = 18.0525;
Location.altitude = 28;


%contratto consumi
   
    %tariffa unica=1; tariffa per fasce=0-->numero fasce;
    tariffa=3;
    
    %mensilità in boletta
    lenght_periodo=2;
  
    %mesi inizio peridi
    mese_start=1;
    mesi(1)=mese_start;
    
    for i=2:12
        mesi(i)=mese_start+i-1;
        if mesi(i)>12
            mesi(i)=mesi(i)-12;
        end
    end
    
   
    consumi_import=[69 94 94 160 97 126 153 97 119 67 48 73 78 77 92 111 91 104];
    
    consumi_lenght=(12/lenght_periodo);
    for i=1:(consumi_lenght)
        t=(i-1)*3+1;
        consumi(i,1)=consumi_import(t);
        consumi(i, 2)=consumi_import(t+1);
        consumi(i, 3)=consumi_import(t+2);
    end
    
    %costo
    costo=[0.1358 0.1144 0.1144 0.1366 0.1145 0.1145 0.1363 0.1144 0.1144  0.1364 0.1145 0.1145 0.1360 0.1137 0.1137 0.1357 0.1139 0.1139];
    
        consumi_lenght=(12/lenght_periodo);
    for i=1:(consumi_lenght)
        t=(i-1)*tariffa+1;
        for l=1:tariffa
        costi(i,l)=costo(t+l-1);
        end
    end
    
    costi(:,2)=costi(:,2)*1.1;
    
    
    %contratto pv system
        pv=0;
        
        irr_mensile_pv= [1168.44 1123.30 918.61 1189.31 1770.88 2938.81 3291.75 3633.41 4047.06 3470.79 2396.22 1715.81];
        eff=0.1;
        prod_mensile_pv=irr_mensile_pv'*eff;
        

        %prezzo_vendita=[0.1 0.1 0.1;0.1 0.1 0.1;0.007 0.008 0.009;0.1 0.1 0.11;0.11 0.11 0.09;0.09 0.09 0.09];
        
