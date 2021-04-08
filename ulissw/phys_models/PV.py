import requests
import pvlib
import numpy as np
from ..utils import read_apikey

CONSTANTS = {
    'n_air' : 1.0002926,
    'n_glass' : 1.58992,
    'noct' : 50.0, # Nominal operating cell temp
    'tc_ref' : 25., # Temp ref for the cell
    'g_ref' : 1000., # Irradiance ref
    'g_noct' : 800., # Irradiance at noct
    't_noct' : 20.,
    'n_diode' : 1.0134, # diode ideality factor  
    'k_boltz' : 8.6173324e-5
}


class PV:
    __BASE_URL = 'https://api.solcast.com.au/world_radiation/forecasts.json'
    __API_KEY = read_apikey('apikey.txt')
    def __init__(self, l=None, k=None, albedo=None, beta=30, gamma=0, 
                 n_series=None, i_sc_ref=None, v_oc_ref=None, t_v_oc=None, t_i_sc=None):
        self.l = l # panel glass thickness [m]
        self.k = k # panel glass extinction factor [1/m]
        self.albedo = albedo
        self.data = None
        self.beta = np.radians(beta); 
        self.gamma = np.radians(gamma);
        self.n_series = n_series
        self.i_sc_ref = i_sc_ref
        self.v_oc_ref = v_oc_ref
        self.t_v_oc = t_v_oc
        self.t_i_sc = t_i_sc
        
    def __query_api(self, lat, long, time_window):
        PARAMS = {'latitude' : lat,
                  'longitude' : long,
                  'hours' : time_window,
                  'api_key' : self.__API_KEY}
        r = requests.get(url = self.__BASE_URL, params = PARAMS)
        return r.json()['forecasts']
    
    def __calc_tau(self, theta, theta_r):
        t_diff = theta_r - theta
        t_sum = theta_r + theta

        tau = np.exp(-self.k*self.l/np.cos(theta_r))
        tau *= (1-1/2*(((np.sin(t_diff))**2/(np.sin(t_sum))**2+(np.tan(t_diff))**2/(np.tan(t_sum))**2)))
        
        return tau
    
    def __calc_irradiance(self, data=None):
        alt_sol = np.array([np.radians(90-point["zenith"]) for point in data])
        alpha = np.array([np.radians(point["azimuth"]-180) for point in data])

        theta = np.arccos(np.cos(alt_sol)*np.cos(alpha-self.gamma)*np.sin(self.beta)+np.sin(alt_sol)*np.cos(self.beta))
        theta_r = np.arcsin(CONSTANTS['n_air']/CONSTANTS['n_glass']*np.sin(theta))
        theta_equiv_diff = np.radians(59.7-0.1388*np.degrees(self.beta)+ 0.001497*np.degrees(self.beta)**2)
        theta_r_equiv_diff = np.arcsin(CONSTANTS['n_air']/CONSTANTS['n_glass']*np.sin(theta_equiv_diff))

        g_ghi = np.array([point["ghi"] for point in data])
        g_dni = np.array([point["dni"] for point in data])
        g_dhi = np.array([point["dhi"] for point in data])
        g_ref = pvlib.irradiance.get_ground_diffuse(np.degrees(self.beta),
                                                    g_ghi, self.albedo, 'urban')
        g_tot =  g_dni*np.cos(theta) + g_dhi*(1+np.cos(self.beta))/2 + g_ref*(1-np.cos(self.beta))/2; 
        
        tau_b = self.__calc_tau(theta, theta_r)
        tau_d = self.__calc_tau(theta_equiv_diff, theta_r_equiv_diff)
        # maybe add g_ref
        g = tau_b*g_dni*np.cos(theta) + tau_d*g_dhi*(1+np.cos(self.beta))/2 + g_ref*(1-np.cos(self.beta))/2

        return g, g_tot
    
    def calc_power(self, lat=45.464211, long=9.191383, time_window=2):
        #data = self.__query_api(lat, long, time_window)
        data = dati['forecasts'][-4:]
        
        g, g_tot = self.__calc_irradiance(data=data)
        # NOCT formula for cell temp
        t_amb = np.array([point["air_temp"] for point in data]) 
        t_amb_ref = CONSTANTS['tc_ref'] - (CONSTANTS['noct'] - CONSTANTS['t_noct'])*CONSTANTS['g_ref']/CONSTANTS['g_noct']
        
        tc = t_amb + (CONSTANTS['noct'] - CONSTANTS['t_noct'])*g_tot/CONSTANTS['g_noct'] 
        
        # thermal voltage
        vt = CONSTANTS['n_diode']*CONSTANTS['k_boltz']*(t_amb+273.15)*self.n_series
        vt_ref = CONSTANTS['n_diode']*CONSTANTS['k_boltz']*(t_amb_ref+273.15)*self.n_series
        
        i0_ref = self.i_sc_ref/(np.exp(self.v_oc_ref/(self.n_series*vt_ref)) - 1)
        
        tc_diff = tc - CONSTANTS['tc_ref']
        i_pv = (g_tot/CONSTANTS['g_ref'])*self.i_sc_ref*(1 + self.t_i_sc*(tc_diff))
        v_oc = self.v_oc_ref*(1 + self.t_v_oc*(tc_diff)) + (self.n_series*vt*CONSTANTS['n_diode'])*np.log(g_tot/CONSTANTS['g_ref'])
        
        i0 = i_pv/(np.exp(v_oc/(self.n_series*vt)) - 1)
        
        V = np.linspace(0, v_oc, 1000, axis=0)
        I = i_pv - i0*(np.exp(V/(self.n_series*vt)) - 1)
        P = I*V        
        p_out = np.max(P, axis=0)
        
        return p_out, I, V