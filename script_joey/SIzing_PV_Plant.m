clear all
%Data from Panel Datasheet (Monocrystalline SunFields)
Panel_lenght= 1.66; %meters
Panel_width= 0.99; %meters
Power_max=300;
%Data from inverter Datasheet (Refusol 100K)
V_min_DC= 460;
V_max_DC=850;
I_max=240;
P_max=115e3;
I_SC_STC=9.97;
V_OC_STC=39.4;
I_MPP=240;
V_MPP=31.2;
%Temperature coefficients for NOCT formula
T_P_max=-0.4/100;
T_V_OC=-0.29/100;
T_I_SC=0.05/100;
%Data from available space 
Rooftop_lenght= 150;
Rooftop_width= 80;
%Design choice according to location or regulation 
Panels_tilt= 33*pi/180;
shading_angle=15*pi/180;
Distance_edge=1.5;
Min_T= -10;
Max_T=60;
%% Number of panels 
Panels_x=floor(( Rooftop_width-(Distance_edge*2))/Panel_width);
Panel_projected=Panel_lenght*cos(Panels_tilt);
Shading_space=(Panel_lenght*sin(Panels_tilt))/tan(shading_angle);
Panels_y=floor((Rooftop_lenght-(Distance_edge*2))/(Panel_projected+Shading_space));
Total_panels=round(Panels_x*Panels_y,0);
%% Theoretical Power Installed
P_inst= Total_panels*Power_max;
%% Modules connected in series
V_OC_Min_T=V_OC_STC*[1+T_V_OC*(Min_T-20)];
Panels_series=floor(V_max_DC/V_OC_Min_T);
%% Strings connected in parallel
I_SC_Max_T=I_SC_STC*[1+T_I_SC*(Max_T-20)];
Parallel_strings=floor(I_max/I_SC_Max_T);
%% Maximum number of modules per inverter
Modules_inverter_max=floor(P_max/Power_max);
Parallel_strings_inverter=floor(Modules_inverter_max/Panels_series);
Panels_parallel=min(Parallel_strings_inverter,Parallel_strings);
Panels_inverter=Panels_parallel*Panels_series;
Inverters=floor((Panels_x*Panels_y)/Panels_inverter);
P_installed=Inverters*Panels_inverter*Power_max;
%%
filename2= 'IDLT_AV_96_N';
IRR_AV_MONTH= xlsread(filename2);
filename3='IRR_ECONOMIC';
IRR_AV_MONTH_ECO= xlsread(filename3);
filename8= 'MGP2016';
MGP2016 = xlsread(filename8);
MGP2016= reshape(MGP2016,[],1);
load('load_profiles.mat');
P_load = 25e3; %25 kW
rend_PV = 0.8;%efficiency of power electronics

start_time = 4;
end_time = 18;

P_pv = @(x,k,y,eta) max(0,eta*(k(3).*x.^2 + k(2).*x + k(1)).*y);%Power produced by the PV x is the irradiance, y is the nominal power of the plant

k = [-0.0138 0.000898 0;...
    -0.074 0.001 0;...
    -0.0187 0.0012 -0.0000004]; % prima colonna sono monocristal

tech = {'mc-Si','a-Si','CIGS'};
%% Irradiance per hour graph of different technologies
Pn = P_installed/1000; %%nominal power of pv y eta is rend_PV
month_label=['Jan';'Feb';'Mar';'Apr';'May';'Jun';'Jul';'Aug';'Sep';'Oct';'Nov';'Dec']
for i = 1:1:3,
    prod_p = P_pv(IRR_AV_MONTH,k(i,:),Pn,rend_PV);
    figure
    plot(prod_p);
    set(gca,'YLim',[0 Pn])
    set(gca,'XLim',[0 size(IRR_AV_MONTH,1)])
    set(gca,'XTick',[1:4:57])
    set(gca,'XTickLabel',[4:1:18])
    xlabel('t [h]')
    ylabel('P [kW/m^2]')
    title(tech(i))
    legend(month_label)
end
%% LOad profile (residential) We have just this load profile (behaviour)
figure
stairs([residential;residential(end)])
set(gca,'XLim',[1 length(residential)+1]);
set(gca,'XTick',[1,(4*4+1):4*4:96,97]);
set(gca,'XTickLabel',[0:4:24]);
xlabel('t [h]')
ylabel('P [%]')
grid on
%% irradiance is in [5am,7pm] this is according to the data obtained from 2016 
res_load_pu = repmat(residential(start_time*4:1:end_time*4),1,12);
res_load_pu(IRR_AV_MONTH == 0) = zeros(size(find(IRR_AV_MONTH == 0)));
figure
stairs([start_time*4:1:(end_time*4)],res_load_pu)
hold on
stairs([residential;residential(end)].*1.3,'k')
set(gca,'XLim',[1 length(residential)+1]);
set(gca,'XTick',[1,(4*4+1):4*4:96,97]);
set(gca,'XTickLabel',[0:4:24]);
xlabel('t [h]')
ylabel('P [%]')
grid on
hold off
legend(month_label)
res_load = res_load_pu.*P_load./100;
%%
irr = reshape(IRR_AV_MONTH,[],1);%%put irradiance in one column
p_load = reshape(res_load,[],1);
%% self consumption graphs func calcuate max PV nominal to get 
options = optimoptions('fmincon','Algorithm','interior-point','Display','none');
self_cons_thres = [.5 .6 .7 .8 .9 1];% define selfconsumption threshold
for ii = 1:length(tech),
    for jj = 1:length(self_cons_thres),
        tic
        [y(ii,jj), fval, exitflag(ii,jj)] = fmincon(@(x) -x, 1,[],[],[],[],0,[],@(x) mycon(x,p_load,irr,k(ii,:),rend_PV,self_cons_thres(jj)),options);
        etime(ii,jj) = toc;
    end
end
clear fval
month_label=['Jan';'Feb';'Mar';'Apr';'May';'Jun';'Jul';'Aug';'Sep';'Oct';'Nov';'Dec']
tech_sel = 3; %selects the technology to be evaluated 
 self_cons_thres = [.5 .6 .7 .8 .9 1];
self_cons_comp = [.7 .8];% set the threshold you want to analyse 
%plot for tech 1 self cons 0.8
figure
plot(P_pv(irr,k(tech_sel,:),y(tech_sel,find(self_cons_thres == self_cons_comp(1))),rend_PV)./1e3)
hold on
plot(p_load./1e3,'g')
set(gca,'XLim',[1 length(p_load)]);
set(gca,'YLim',[0 1.1*max(max(P_pv(irr,k(tech_sel,:),y(tech_sel,find(self_cons_thres == self_cons_comp(1))),rend_PV)),max(P_pv(irr,k(tech_sel,:),y(tech_sel,find(self_cons_thres == self_cons_comp(1))),rend_PV)))]/1e3)
set(gca,'XTick',[28:57:12*57]);
set(gca,'XTickLabel',month_label);
xlabel('t [h]')
ylabel('P [kW]')
legend('P','Load')
grid on
hold off
%Plot for tech 1 self cons 1 
figure
plot(P_pv(irr,k(tech_sel,:),y(tech_sel,find(self_cons_thres == self_cons_comp(2))),rend_PV)./1e3)
hold on
plot(p_load./1e3,'r')
set(gca,'XLim',[1 length(p_load)]);
set(gca,'YLim',[0 1.1*max(max(P_pv(irr,k(tech_sel,:),y(tech_sel,find(self_cons_thres == self_cons_comp(2))),rend_PV)),max(P_pv(irr,k(tech_sel,:),y(tech_sel,find(self_cons_thres == self_cons_comp(1))),rend_PV)))]/1e3)
set(gca,'XTick',[28:57:12*57]);
set(gca,'XTickLabel',month_label);
xlabel('t [h]')
ylabel('P [kW]')
legend('P','Load')
grid on
hold off
%% Calculate the max energy and power storage for all tech and all self cons thresholds 

for jj = 1:length(self_cons_thres),
    for ii = 1:length(tech)
        a = P_pv(irr,k(ii,:),y(ii,jj),rend_PV) - p_load;
        a(a<0) = zeros(size(find(a<0)));
        max_P_stg(ii,jj) = max(a)/1e3;
        max_E_stg(ii,jj) = max(sum(reshape(a,size(IRR_AV_MONTH)))./4)/1e3;
        clear a
    end
end

%% plot the max power od storage 
figure
plot(max_P_stg','Marker','s','MarkerSize',7,'LineStyle','none')
legend(tech)
set(gca,'XTick',[1:1:length(self_cons_thres)])
set(gca,'XTickLabel',self_cons_thres)
xlabel('self-consumption threshold [p.u.]')
ylabel('P [kW]')
grid on
%% plot the max energy of storage 
figure
plot(max_E_stg','Marker','s','MarkerSize',7,'LineStyle','none')
legend(tech)
set(gca,'XTick',[1:1:length(self_cons_thres)])
set(gca,'XTickLabel',self_cons_thres)
xlabel('self-consumption threshold [p.u.]')
ylabel('E [kWh]')
grid on


%% change k for tech 
res_load_pu_eco = repmat(residential,1,12);
res_load_eco= res_load_pu_eco.*P_load./100;
irr_eco = reshape(IRR_AV_MONTH_ECO,[],1);%%put irradiance in one column
p_load_eco = reshape(res_load_eco,[],1);
prod_p_eco=P_pv(IRR_AV_MONTH_ECO,k(3,:),Pn,rend_PV);% change k for technology 