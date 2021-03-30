import requests
import pvlib
import numpy as np


CONSTANTS = {
    'n_air' : 1.0002926,
    'n_glass' : 1.58992,
    'NOCT' = 50.
}


class PV:
    BASE_URL = 'https://api.solcast.com.au/world_radiation/forecasts.json'
    API_KEY = 'YOUR_API_KEY'
    def __init__(self, l=None, k=None, albedo=None, beta=30, gamma=0):
        self.l = l
        self.k = k
        self.albedo = albedo
        self.data = None
        self.beta = np.radians(beta); 
        self.gamma = np.radians(gamma);
    
    def __query_api(self, lat, long, time_window):
        PARAMS = {'latitude' : lat,
                  'longitude' : long,
                  'hours' : time_window,
                  'api_key' : self.API_KEY}
        r = requests.get(url = self.BASE_URL, params = PARAMS)
        return r.json()['forecasts']
    
    def __calc_tau(self, theta, theta_r):
        t_diff = theta_r - theta
        t_sum = theta_r + theta

        tau = np.exp(-self.k*self.l/np.cos(theta_r))
        tau *= (1-1/2*(((np.sin(t_diff))**2/(np.sin(t_sum))**2+(np.tan(t_diff))**2/(np.tan(t_sum))**2)))
        
        return tau
    
    def __calc_irradiance(self, lat=45.464211, long=9.191383, 
                          time_window=2, data=None):
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
        
        g = tau_b*g_dni*np.cos(theta) + tau_d*g_dhi*(1+np.cos(self.beta))/2 + g_ref*(1-np.cos(self.beta))/2

        return g
    
    def calc_power(self):
        data = self.__query_api(lat, long, time_window)
        
        g = self.__calc_irradiance(data=data)