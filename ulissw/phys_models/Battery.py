import os
from ..datasets import EtaParser
from ..models import EtaRegressor

class Battery:
    def __init__(self, soc=0, vn=None, n_cycles=0, 
                 max_kwh=None, pn=None, eff_data=None,
                 static_eta=None):
        self.soc = soc # percentage of charge
        self.v_n = vn # nominal voltage [V]
        self.n_cycles = n_cycles # integer n.of cycles
        self.max_kwh = max_kwh # battery capacity [kWh]
        self.p_n = pn # nominal power [kW]
        self.e_max = self.max_kwh*3.6e3 #prima era 3.6e6
        self.calc_eta = True 
        if isinstance(static_eta, float):  # whether to use a static efficiency
            self.calc_eta = False
            self.static_eta = static_eta
        
        if eff_data is None:
            my_path = os.path.abspath(os.path.dirname(__file__))
            eff_data  = os.path.join(my_path, '../../data/eta_data.csv')
            
        self.eff_data = EtaParser(path=eff_data)
        self.eff_model = self.__infer_efficiency_model()

    def reset(self, soc=0):
        self.soc = soc

    def charge(self, p_in, dt):
        old_soc = self.soc
        
        if p_in >= self.p_n:
            p_charge = self.p_n
            e_left = (p_in - self.p_n)/3.6e3*dt
        else:
            p_charge = p_in
            e_left = 0
        
        # charging energy
        e_ch = p_charge*dt
        new_soc = self.soc + e_ch/self.e_max
                    
        if new_soc > 1:
            over_soc = new_soc - 1
            over_e = self.e_max*over_soc
            e_left += over_e
            # p_left = over_e/dt
            new_soc = 1
        self.soc = new_soc    
        self.n_cycles += (self.soc - old_soc)
        
        return e_left
    
    def __infer_efficiency_model(self):
        X, y = self.eff_data.parse_data()
        model = EtaRegressor(X, y)        
        _ = model.fit_data()

        return model
    
    def max_load(self):
        if self.calc_eta:
            X = self.eff_data.format(1, self.soc)
            eta = self.eff_model.predict(X)
        else:
            eta = self.static_eta        
        e_dis = self.soc*self.max_kwh*eta
        
        return e_dis
    
    def free_storage(self):
        return (1-self.soc)*self.max_kwh/self.static_eta

    def empty(self, return_energy=False):
        if return_energy:
            e_dis = self.max_load()
            self.soc = 0
            return e_dis
        self.soc = 0

    def discharge(self, p_out, dt, testing=False):
        if not self.static_eta:
            X = self.eff_data.format(p_out/self.p_n, self.soc)
            eta = self.eff_model.predict(X)
        else:
            eta = 1
        
        if p_out/eta > self.p_n:
            p_draw = self.p_n
            e_left = (p_out - self.p_n)/3.6e3*dt
        else:
            p_draw = p_out/eta
            e_left = 0
        
        e_dis = p_draw*dt
        old_soc = self.soc
        
        new_soc = self.soc - e_dis/self.e_max

        if new_soc < 0:
            over_soc = -new_soc
            over_e = self.e_max*over_soc
            e_left += over_e/3.6e3
            # p_left = over_e/dt
            new_soc = 0

        self.soc = new_soc    
        if testing:
            print(f"[!!!] Eta estimated at {eta:.4f} [!!!]")
            return e_left, eta
        return e_left

    @staticmethod
    def model_info():
        print("[|||] This class models a battery for energy storage [|||]"+
              "\nParameters to initialize (they're all public members):"+
              "\n\t-StateOfCharge -> Percentage of charge, default:0"+
              "\n\t-n_cycles -> N. of charge/discharge cycles, integer number, default:0"+
              "\n\t-v_n -> Nominal Voltage, in Volt [V], default:None, example:700"+
              "\n\t-p_n -> Nominal Power, in kilowatts [kW], default:None, example:250"+
              "\n\t-max_kwh -> Capacity of the battery in kilowatts per hour, [kWh], default:None, example:50"+
              "\n\nInstructions for the methods:"+
              "\n\t-charge(p_in, dt) -> Apply an amount of power p_in [kw] for a "+
              "timespan of dt [s].\n\tReturns the power which could not be stored, in terms of "+
              "energy over the specified timespan\n"+
              "\n\n\t-discharge(p_out, dt) -> Draw an amount of power p_out [kw] for a "+ 
              "timespan of dt [s].\n\tReturns the power which was not available, in terms of energy "+
              "over the specified timespan")
        