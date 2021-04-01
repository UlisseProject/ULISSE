class Battery:
    def __init__(self, soc=0, vn=None, n_cycles=0, 
                 max_kwh=None, pn=None):
        self.soc = soc
        self.v_n = vn
        self.n_cycles = n_cycles
        self.max_kwh = max_kwh
        self.p_n = pn
        self.e_max = self.max_kwh*3.6e6
        
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
    
    def __calc_efficiency(self):
        # TODO : regression from eta data
        pass
    
    def discharge(self, p_out):
        eta = self.__calc_efficiency()
        
        e_dis = p_out*dt/eta