import pandas as pd
import os

class ConsumptionProfile:
    def __init__(self, folder):
        self.folder = folder
        self.df = self.load_data(folder)
    
    @staticmethod
    def load_data(path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f'The specified {path} does not point to an existing file')
        
        df = pd.read_csv(path, sep=";", header=None)
        df.drop([0,1, 4,5, 6], axis=1, inplace=True)
        df.rename(columns={2:"date_time", 3:"kwh_hh"}, inplace=True)
        #df['date'], df['time'] = zip(*df.apply(lambda x: (x.date_time.split(" ")[0], x.date_time.split(" ")[1]), axis=1))
        df['kwh_hh'] = df['kwh_hh'].map(lambda x: float(x.replace(",", ".")))
        #df['kwh'] = df['kwh_hh']*2
        df['kwh'] = df['kwh_hh']
        df.drop(['kwh_hh'], axis=1, inplace=True)
        return df