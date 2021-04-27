import requests
import pvlib
import numpy as np
import pandas as pd
import os
import datetime
from dateutil import parser
from ..utils import read_apikey


CONSTANTS = {
    'n_air' : 1.0002926,
    'n_glass' : 1.58992,
    'noct' : 50.0, # Nominal operating cell temp
    'tc_ref' : 25., # Temp ref for the cell
    'g_ref' : 1000., # Irradiance ref
    'g_noct' : 800., # Irradiance at noct
    't_noct' : 20.,
    'n_diode_old' : 1.0134, # diode ideality factor  
    'n_diode' : 1.259, # diode ideality factor  
    'k_boltz' : 8.6173324e-5,
}


class PV:
    __BASE_URL = 'https://api.solcast.com.au/world_radiation/forecasts.json'
    __API_KEY =  'SET_BELOW'
    def __init__(self, l=None, k=None, albedo=None, beta=30, gamma=0, 
                 nc_series=None, nc=None, i_sc_ref=None, v_oc_ref=None, 
                 t_v_oc=None, t_i_sc=None, power_resolution=1000):
        self.l = l # panel glass thickness [m]
        self.k = k # panel glass extinction factor [1/m]
        self.albedo = albedo
        self.beta = np.radians(beta); 
        self.gamma = np.radians(gamma);
        self.nc_series = nc_series # number of cells series-connected
        self.nc = nc # total number of cells
        self.i_sc_ref = i_sc_ref
        self.v_oc_ref = v_oc_ref
        self.t_v_oc = t_v_oc
        self.t_i_sc = t_i_sc
        self.power_resolution = power_resolution
        self.data = None
        self.static_data_path = None
        self.data_filter = None

        if self.__API_KEY == 'SET_BELOW':
            my_path = os.path.abspath(os.path.dirname(__file__))
            path = os.path.join(my_path, "../../apikey.txt")
            self.__API_KEY = read_apikey(path)

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
        zenith = self.__get_data_column('zenith')
        azimuth = self.__get_data_column('azimuth')

        alt_sol = np.radians(90 - zenith)
        alpha = np.radians(azimuth - 180)

        theta = np.arccos(np.cos(alt_sol)*np.cos(alpha-self.gamma)*np.sin(self.beta)+np.sin(alt_sol)*np.cos(self.beta))
        theta_r = np.arcsin(CONSTANTS['n_air']/CONSTANTS['n_glass']*np.sin(theta))
        theta_equiv_diff = np.radians(59.7-0.1388*np.degrees(self.beta)+ 0.001497*np.degrees(self.beta)**2)
        theta_r_equiv_diff = np.arcsin(CONSTANTS['n_air']/CONSTANTS['n_glass']*np.sin(theta_equiv_diff))
        
        g_ghi = self.__get_data_column('ghi')
        g_dni = self.__get_data_column('dni')
        g_dhi = self.__get_data_column('dhi')
        g_ref = pvlib.irradiance.get_ground_diffuse(np.degrees(self.beta),
                                                    g_ghi, self.albedo, 'urban')
        g_tot =  g_dni*np.cos(theta) + g_dhi*(1+np.cos(self.beta))/2 + g_ref*(1-np.cos(self.beta))/2

        tau_b = self.__calc_tau(theta, theta_r)
        tau_d = self.__calc_tau(theta_equiv_diff, theta_r_equiv_diff)
        # maybe add g_ref
        g = tau_b*g_dni*np.cos(theta) + tau_d*g_dhi*(1+np.cos(self.beta))/2 + g_ref*(1-np.cos(self.beta))/2

        return g, g_tot
    
    def calc_power(self, lat=45.464211, long=9.191383, time_window=2):
        if self.static_data_path is None:
            #data = dati['forecasts'][-4:]
            self.data = self.__query_api(lat, long, time_window)
        else:
            self.data = self.__import_from_csv(self.static_data_path)
        
        g, g_tot = self.__calc_irradiance()
        # NOCT formula for cell temp
        t_amb = self.__get_data_column('air_temp')
        t_amb_ref = CONSTANTS['tc_ref'] - (CONSTANTS['noct'] - CONSTANTS['t_noct'])*CONSTANTS['g_ref']/CONSTANTS['g_noct']
        
        tc = t_amb + (CONSTANTS['noct'] - CONSTANTS['t_noct'])*g_tot/CONSTANTS['g_noct'] 
        tc_ref = t_amb_ref + (CONSTANTS['noct'] - CONSTANTS['t_noct'])*g_tot/CONSTANTS['g_noct'] 
        
        # thermal voltage
        vt = CONSTANTS['n_diode']*CONSTANTS['k_boltz']*(tc+273.15)
        vt_ref = CONSTANTS['n_diode']*CONSTANTS['k_boltz']*(tc_ref+273.15)
        
        i0_ref = self.i_sc_ref/(np.exp(self.v_oc_ref/(self.nc_series*vt_ref)) - 1)
        
        tc_diff = tc - CONSTANTS['tc_ref']
        i_pv = (g_tot/CONSTANTS['g_ref'])*self.i_sc_ref*(1 + self.t_i_sc*(tc_diff))
        v_oc = self.v_oc_ref*(1 + self.t_v_oc*(tc_diff)) + (self.nc*vt*CONSTANTS['n_diode'])*np.log(g_tot/CONSTANTS['g_ref'])
        
        i0 = i_pv/(np.exp(v_oc/(self.nc_series*vt)) - 1)
        
        v_oc[v_oc == -np.inf] = 0
        V = np.linspace(0, v_oc, self.power_resolution, axis=0)
        I = i_pv - i0*(np.exp(V/(self.nc_series*vt)) - 1)

        P = I*V        
        p_out = np.max(P, axis=0)
        
        return p_out

    def set_static_data_path(self, path):
        if os.path.isfile(path):
            self.static_data_path = path
        else:
            raise FileNotFoundError('The specified path does not point to an existing file')

    def disable_static_data_path(self):
        self.static_data_path = None
        self.data_filter = None

    def filter_static_doy_hour(self, filter_path):
        if self.disable_static_data_path is None:
            raise Exception("You need to specify a path for static data if you want to filter them")
        if not os.path.isfile(filter_path):
            raise FileNotFoundError('The specified path does not point to an existing file')
        # data/data_pv.csv
        df_filter = pd.read_csv(filter_path, delimiter=",") 
        df_filter.columns = df_filter.columns.map(str.lower)
        if not {'doy', 'hour', 'poutdc'}.issubset(set(df_filter.columns)):
            raise ValueError("The filter dataframe should contain columns 'doy' and 'hour'")
        self.data_filter = df_filter

    def __import_from_csv(self, path):
        # data/45.502941_9.156574_PVsyst_PT60M.csv
        test_data = pd.read_csv(path, delimiter=",")
        test_data = PV.__convert_history_data_iface(test_data, path)
        if self.data_filter is not None:
            test_data = self.__merge_doy_hour_filter(test_data)
        return test_data

    def __merge_doy_hour_filter(self, df):
        dfh = self.data_filter[['doy', 'hour', 'poutdc']]
        merged_df = df.merge(dfh, on=['doy', 'hour'], how='inner')
        
        return merged_df

    def __get_data_column(self, key):
        if self.static_data_path is not None:
            data = self.data[key].to_numpy(dtype=np.float64) 
        else:
            data = np.array([point[key] for point in self.data])

        return data

    @staticmethod
    def __convert_history_data_iface(df, path):
        if 'PVsyst' in path:
            conv_dict = {'tamb': 'air_temp'}
            droppables = ['cloudopacity', 'minute']

            df.columns = df.columns.map(str.lower)
            df.drop(columns=droppables, inplace=True)        
            df.columns = df.columns.map(lambda x: conv_dict.get(x, x))
            df['doy'] = df[['year', 'month', 'day']].apply(PV.__map_ymd_to_doy, axis=1)
        else:
            conv_dict = {'airtemp': 'air_temp'}
            droppables = ['albedodaily', 'cloudopacity', 'period', 'periodstart']

            df.columns = df.columns.map(str.lower)
            df.drop(columns=droppables, inplace=True)           
            df.columns = df.columns.map(lambda x: conv_dict.get(x, x))
            df['year'], df['month'], df['day'], df['hour'], df['min'] = zip(*df.apply(PV.__map_iso_to_ymd, axis=1))
        return df
    
    @staticmethod
    def __map_iso_to_ymd(x):
        date_obj = parser.isoparse(x.periodend).timetuple()
        return date_obj.tm_year, date_obj.tm_mon, date_obj.tm_mday,  date_obj.tm_hour, date_obj.tm_min

    @staticmethod
    def __map_ymd_to_doy(x):
        return datetime.date(x.year, x.month, x.day).timetuple().tm_yday

    @staticmethod
    def model_info():
        print("[|||] This class models a PV  [|||]"+
              "\nParameters to initialize (they're all public members):"+
              "\n\t-l -> panel glass thickness [m]"+
              "\n\t-k -> panel glass extinction factor [1/m]"+
              "\n\t-nc -> integer number, total num of cells of the module"+
              "\n\t-nc_series -> integer number, total num of cells connected in series "+
              "\n\t-albedo -> "+
              "\n\t-beta -> [degrees]"+
              "\n\t-gamma -> [degrees]"+
              "\n\t-i_sc_ref -> []"+
              "\n\t-v_oc_ref -> []"+
              "\n\t-t_v_oc -> []"+
              "\n\t-t_i_sc -> []"+
              "\n\nInstructions for the methods:"+
              "\n\t-set_static_data_path(path) -> Ask the model to get data from a static file at the"+ 
              "specified path.\n\tThe model will not use the API and instead use the file which should contain "+
              "irradiance and temp data in a csv file\n. To disable this behavior call 'disable_static_data_path'"+
              "\n\n\t-filter_static_doy_hour(path) -> This method can be used only after setting a static data path to "+
              "specify\na filter on the day of year/hour data point that should be considered. DoY/Hour not contained "+
              "in the file hereby specified will be discarded"+
              "\n\n\t-calc_power(lat, long, time_window) -> Asks the model to query the weather API at the specified "+ 
              "location for the given window\n\tand evaluate power output of the model. If static data are provided, "+
              "they will be used instead and there is no need to pass lat, long or time.")