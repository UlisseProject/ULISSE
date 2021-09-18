from init_scripts import init_script
init_script()

from ulissw.optim import PVBoptimizer
from ulissw.phys_models import PV, Battery
from ulissw.datasets import CustomerDataset
import numpy as np
import random

random.seed(43)
np.random.seed(43)

print("[|||] Loading consumption dataset... [|||]")
ds = CustomerDataset('data/sonnen', prediction_interval=(192, 16), strategy='group20_sum')
ds.min_max()

############### PV model parameters
T_V_OC = -2.9e-3  
T_I_SC = 5e-4
I_sc_ref = 9.73
V_oc_ref = 39.2

l = 3e-3 # panel glass thickness [m]
k = 4    # panel glass extinction factor [1/m]
albedo=0.1
gamma=-6.5
nc=60
nc_series=60
###############
print("[|||] Building PV model... [|||]")
# Instantiate PV class with parameter
pv = PV(l=l, k=k, albedo=albedo, nc=nc, nc_series=nc_series, 
        t_v_oc=T_V_OC, t_i_sc=T_I_SC, gamma=gamma,
        i_sc_ref=I_sc_ref, v_oc_ref=V_oc_ref)

# Set pv class to take data from Solcast history instead of query API
pv.set_static_data_path('data/corridoni_45.46345540118191_9.203073624968606_Solcast_PT5M.csv')
# filter irradiance data with timestamps from the dataset
filter_df = ds.get_timestamp_filter()
pv.filter_static_doy_hour(filter_df)
# obtain power prediction of our model
p_out = pv.calc_power()
pv.data['p_prod'] = p_out

######## create battery model
print("[|||] Building Storage model... [|||]")
base_battery = {
    'cap': 14, # kWh
    'pn': 5*0.85 # kW
}
n_base = 14
max_kwh = base_battery['cap']*n_base
pn = base_battery['pn']*n_base
battery = Battery(soc=0, vn=700, n_cycles=0, static_eta=0.9,
                  max_kwh=max_kwh, pn=pn)
######################


####### Build optimizer
print("[|||] Building optimizer... [|||]")
model_path = 'logs/group20_sum/WAPE/TCNTrainerepoch31.pth'
ds_path = 'data/sonnen'
n_panels = 100

pvb = PVBoptimizer(pv=pv, battery=battery,  
                   prices_path="data/bands_prices.csv", 
                   n_panels=n_panels,
                   model_path=model_path, 
                   dataset_path=ds_path,
                   price_delta_inflate=0)

# run optimizer with ULISSE's strategy on a full year
spent_ul, charged_ul, soc_ul, \
    cons_ul, prod_ul = pvb.run_ulisse(days=list(range(364)))

# plot outcomes on the days of the year from 120 to 125
start_day = 120 # day of year
end_day = 125
pvb.plot_outcomes(days=range(start_day, end_day), save='', behavior='ulisse')

# run optimizer without ULISSE to compare outcomes
spent, _, _, _, _ = pvb.run_standard_behavior(days=list(range(364)))

print(f'In total you have spent {spent_ul:.2f} €, with a saving of' + 
      f' {(spent-spent_ul):.2f} € compared to a standard system')
