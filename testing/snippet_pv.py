from ulissw.phys_models import PV

## old values
I_sc_ref = 8.48 
V_oc_ref = 37.10  

## current values
T_V_OC = -2.9e-4 
T_I_SC = 5e-4
I_sc_ref = 9.73 
V_oc_ref = 39.2 
l = 3e-3
k = 4
albedo=0.1
nc=60
nc_series=20
gamma = -6.5
pv = PV(gamma=gamma, l=l, k=k, albedo=albedo, nc=nc, nc_series=nc_series, t_v_oc=T_V_OC, t_i_sc=T_I_SC, 
        i_sc_ref=I_sc_ref, v_oc_ref=V_oc_ref)

### To get experimental data and the history from Solcast
pv.set_static_data_path('data/45.502941_9.156574_PVsyst_PT60M.csv')
pv.power_resolution = 10
pv.filter_static_doy_hour('data/data_pv.csv')
####

p_out = pv.calc_power(time_window=8)