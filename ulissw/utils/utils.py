import os
import glob
from dateutil import parser


class RangeDict:
    def __init__(self, default=None):
        self.dic = {}
        self.default = default
    
    def get(self, key, default=None):
        val = self[key]
        if val != self.default:
            return val
        if default is None:
            return self.default
        return default
    
    def __getitem__(self, key):
        for k in self.dic:
            if key in k:
                return self.dic[k]
        return self.default
        
    def __setitem__(self, key, val):
        self.dic[key] = val


def get_band_price(x, bands, df_prices):
    date_obj = parser.parse(x.date_time, dayfirst=True).timetuple()
    wday = date_obj.tm_wday
    band = bands[wday][x.hour]
    price = df_prices.loc[x.month, band]
    return price


def read_apikey(path):
    with open(path, "r") as f:
    	line=f.readline()
    
    	return line.strip()


def get_filepath(fname):
    os.makedirs('imgs', exist_ok=True)
    if os.path.isfile(os.path.join('imgs', fname +".eps")):
        n_file = 1+ max(list(
                    map(lambda x: int(x), 
                    map(lambda x: x if not (x.islower()) or (x.isupper()) else 0, 
                    map(lambda x: x.split(".eps")[0].split("_")[-1], 
                        glob.glob("imgs/"+fname+"*.eps"))))))
        file_suffix = '_' + str(n_file)
    else:  
        file_suffix = ''
    return os.path.join('imgs', fname + file_suffix + ".eps")