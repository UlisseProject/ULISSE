from ulissw.phys_models import PV

T_V_OC = -0.29/100 
T_I_SC = 0.05/100
I_sc_ref = 8.48 # from matlab
V_oc_ref = 37.10  # from matlab

I_sc_ref = 9.73 # from Niccolai's paper
V_oc_ref = 39.2 # from Niccolai's paper
l=2.5e-3
k=10
albedo=0.1
nc=60
nc_series=20
pv = PV(l=l, k=k, albedo=albedo, nc=nc, nc_series=nc_series, t_v_oc=T_V_OC, t_i_sc=T_I_SC, 
        i_sc_ref=I_sc_ref, v_oc_ref=V_oc_ref)

### To get experimental data and the history from Solcast
pv.set_static_data_path('data/45.502941_9.156574_PVsyst_PT60M.csv')
pv.power_resolution = 10
pv.filter_static_doy_hour('data/data_pv.csv')
####

p_out, I, V, g, v_oc, vt, tc_diff,i_pv, g_tot, tc = pv.calc_power(time_window=8)