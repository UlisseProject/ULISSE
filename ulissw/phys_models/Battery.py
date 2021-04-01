from datasets import EtaParser
from models import EtaRegressor

class Battery:
    def __init__(self, soc=0, vn=None, n_cycles=0, 
                 max_kwh=None, pn=None, eff_data='data/eta_data.csv'):
        self.soc = soc
        self.v_n = vn
        self.n_cycles = n_cycles
        self.max_kwh = max_kwh
        self.p_n = pn
        self.e_max = self.max_kwh*3.6e6
        self.eff_data = EtaParser(path=eff_data)
        self.eff_model = self.__infer_efficiency()

    def charge(self, p_in, dt):
        old_soc = self.soc
        e_n = self.pn*dt
        c_n = e_n/self.v_n
        
        # charging energy
        e_ch = p_in*dt
        p_left = 0
        
        if p_in < self.p_n:
            new_soc = self.soc + e_ch/self.e_max
                        
            if new_soc > 1:
                over_soc = new_soc - 1
                over_e = self.e_max*over_soc
                p_left = over_e/dt
                new_soc = 1
            self.soc = new_soc
                
            self.n_cycles += (self.soc - old_soc)
        else
            raise ValueError("Nominal power charge excedeed")
        
        return p_in, p_left
    
    def __infer_efficiency(self):
        X, y = self.EtaParser.parse_data()
        model = EtaRegressor(X, y)        
        _ = model.fit_data()

        return model
    
    def discharge(self, p_out):
        # TODO : replace with predict
        # eta = self.__calc_efficiency()
        
        e_dis = p_out*dt/eta