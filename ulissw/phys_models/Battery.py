from ..datasets import EtaParser
from ..models import EtaRegressor

class Battery:
    def __init__(self, soc=0, vn=None, n_cycles=0, 
                 max_kwh=None, pn=None, eff_data=None):
        self.soc = soc # percentage of charge
        self.v_n = vn # nominal voltage [V]
        self.n_cycles = n_cycles # integer n.of cycles
        self.max_kwh = max_kwh # battery capacity [kWh]
        self.p_n = pn # nominal power [W]
        self.e_max = self.max_kwh*3.6e6
        
        if eff_data is None:
            my_path = os.path.abspath(os.path.dirname(__file__))
            eff_data  = os.path.join(my_path, '../../data/eta_data.csv')
            
        self.eff_data = EtaParser(path=eff_data)
        self.eff_model = self.__infer_efficiency_model()

    def charge(self, p_in, dt):
        old_soc = self.soc
        
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
        else:
            raise ValueError("Nominal power charge excedeed")
        
        return p_left
    
    def __infer_efficiency_model(self):
        X, y = self.eff_data.parse_data()
        model = EtaRegressor(X, y)        
        _ = model.fit_data()

        return model
    
    def discharge(self, p_out, dt, testing=False):
        X = self.eff_data.format(p_out/self.p_n, self.soc)
        eta = self.eff_model.predict(X)
        e_dis = p_out*dt/eta
        
        old_soc = self.soc
        p_left = 0
        if p_out/eta < self.p_n:
            new_soc = self.soc - e_dis/self.e_max

            if new_soc < 0:
                over_soc = -new_soc
                over_e = self.e_max*over_soc
                p_left = over_e/dt
                new_soc = 0
            self.soc = new_soc
        else:
            raise ValueError("Nominal power discharge excedeed")
        
        if testing:
            print(f"[!!!] Eta estimated at {eta:.4f} [!!!]")
            return p_left, eta
        return p_left

    @staticmethod
    def dimensional_info():
        print("[|||] This class models a battery for energy storage [|||]")
        print("Parameters to initializate (they're all public members):")
        print("\t-StateOfCharge -> Percentage of charge, default:0")
        print("\t-n_cycles -> N. of charge/discharge cycles, integer number, default:0")
        print("\t-v_n -> Nominal Voltage, in Volt [V], default:None, example:700")
        print("\t-p_n -> Nominal Power, in kilowatts [kW], default:None, example:250")
        print("\t-max_kwh -> Capacity of the battery in kilowatts per hour, [kWh], default:None, example:50")
        print("\nInstructions for the methods:")
        print("\t-charge(p_in, dt) -> Apply an amount of power p_in [kw] for a "+ 
              "timespan of dt [s].\n\tReturns the power which could not be stored, in terms of "+
              "energy over the specified timespan\n")
        print("\t-discharge(p_out, dt) -> Draw an amount of power p_out [kw] for a "+ 
              "timespan of dt [s].\n\tReturns the power which was not available, in terms of energy "+
              "over the specified timespan")
        