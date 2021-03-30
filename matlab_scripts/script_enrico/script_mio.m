clear all
clc
addpath ..

%run('casa_familiare_torchiarolo.m')
run('casa_familiare_140m2_legnano.m')


run('prezzi_vendita.m')

prezzi_rid=prezzi_rid/1000;
prezzi_rid_fascia=zeros(6,consumi_lenght,3);

for i=1:consumi_lenght
    t=(i-1)*lenght_periodo+1;
    for l=t:(t+lenght_periodo-1)
    prezzi_rid_fascia(:,i,:)=prezzi_rid_fascia(:,i,:)+prezzi_rid(:,l, :);
    end
end

prezzi_rid_fascia=prezzi_rid_fascia/lenght_periodo;

for i=1:consumi_lenght
    for j=1:3
        prezzi_rid_fascia_loc(i,j)=prezzi_rid_fascia(location,i,j);
    end
end
        
%conta che qui stiamo facendo approssimazione pesantissima: stai considerando che la 
%produzione si divide per F1 e F2 ed F3 ma poi in realà i prezzi di vendita variano per festivi
%avresti potuto considerarlo conteggiando i festivi ma hai prefeito di no
%perchè lo stesso problema lo hai sui consumi, e lì non puoi far enulla. In
%questa maniera, siccome letariffe si dovrebbe equiparare, i problemi
%siautobilanciano. La soluzione sta sempre e solo nell'utilizzare dati
%reali e seri.

        

y=2016;

giorni_mese=[31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

for m=1:12
    clear ClearSkyGHI;
for i=1:giorni_mese(m)
% Create 1-min time series for Jan 1, 2012
DN = datenum(y, m,i):1/(24*60):datenum(y, m, i, 23, 59, 59);
Time = pvl_maketimestruct(DN, 1);
[SunAz, SunEl, ApparentSunEl, SolarTime]=pvl_ephemeris(Time, Location);
ApparentZenith = 90-ApparentSunEl;
ClearSkyGHI(:,i) = pvl_clearsky_haurwitz(ApparentZenith);
end

ClearSkyGHI_medio=sum(ClearSkyGHI, 2)/giorni_mese(m);
s=60;
F(m,1)=sum(ClearSkyGHI_medio(8*s:18*s,1))/sum(ClearSkyGHI_medio);
F(m,2)=(sum(ClearSkyGHI_medio(7*s,1))+sum(ClearSkyGHI_medio(19*s:22*s,1)))/sum(ClearSkyGHI_medio);
F(m,3)=(sum(ClearSkyGHI_medio(s:6*s,1))+sum(ClearSkyGHI_medio(23*s:24*s,1)))/sum(ClearSkyGHI_medio);
CSGHI(m)=(sum(sum(ClearSkyGHI)))/60/1000;
end

if pv==0
    for m=1:12
        for fascia=1:3
            produzione_mensile_pv_fasce(m, fascia)=F(m,fascia)*prod_mensile_pv(m);
        end
    end
else    
end

produzione_period_pv_fasce=zeros(consumi_lenght, 3);

for i=1:consumi_lenght
        t=(i-1)*lenght_periodo+1;
        for l=1:lenght_periodo
        produzione_period_pv_fasce(i,:)=produzione_period_pv_fasce(i,:)+produzione_mensile_pv_fasce(t+l-1, :);
        end
        produzione_period_pv_fasce(i,:)=produzione_period_pv_fasce(i,:);
end




%contratto solo acquisto
spesa_fascia_acquisto=costi(:,:).*consumi(:, :);
spesa_totale_acquisto=sum(spesa_fascia_acquisto(:,:), 2);


%contratto vendita-acquisto indipendenti

    for i=1:consumi_lenght
            spesa_fascia_ind(i, :)=costi(i,:).*consumi(i, :);
            spesa_totale_ind(i)= sum(spesa_fascia_ind(i, :));
            
            incasso_fascia_ind(i, :)=prezzi_rid_fascia_loc( i, :).*produzione_period_pv_fasce(i,:);
            incasso_totale_ind(i)=sum(incasso_fascia_ind(i, :));
            
            netto_ind(i)=-incasso_totale_ind(i)+spesa_totale_ind(i);
    end
    
    
    %contratto solo consumo, classico e con storage
    
    for i=1:consumi_lenght
        for fascia=1:3
           deficit_energetico(i, fascia)=consumi(i, fascia)-produzione_period_pv_fasce(i,fascia); %per ogni fascia e per ogni bimestre quanto comprare e quanto vendere
            
           if deficit_energetico(i, fascia)>0
               spesa_fascia_solo_consumo(i, fascia)=deficit_energetico(i, fascia)*costi(i, fascia);
               netto_fascia_classico(i, fascia)=spesa_fascia_solo_consumo(i, fascia);
               netto_fascia_storage(i, fascia)=deficit_energetico(i, fascia)*costi(i, 3);
            else
                spesa_fascia_solo_consumo(i, fascia)=0;
                netto_fascia_classico(i, fascia)=deficit_energetico(i, fascia)*prezzi_rid_fascia_loc(i, fascia);
                netto_fascia_storage(i, fascia)=deficit_energetico(i, fascia)*prezzi_rid_fascia_loc(i, 1); 
            end
        end
        
            spesa_totale_solo_consumo(i)=sum(spesa_fascia_solo_consumo(i,:));
            netto_totale_classico(i)=sum(netto_fascia_classico(i, :));
            netto_totale_storage(i)=sum(netto_fascia_storage(i, :)); %netto_fascia_storage bolletta bimestre per bimestre ma divisa in fasce
    end
    
comparazione_contratti(:,1)=spesa_totale_acquisto;
comparazione_contratti(:,2)=netto_ind';
comparazione_contratti(:,3)=spesa_totale_solo_consumo';
comparazione_contratti(:,4)=netto_totale_classico'; 
comparazione_contratti(:,5)=netto_totale_storage'; %totali della bolletta bimestre per bimestre

totali_comparazione=sum(comparazione_contratti(:,:), 1);
