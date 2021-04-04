# Battery Testing 
# - to be executed in a sibling folder of 'ulissw', or the imports below
# won't work
from ulissw.phys_models import Battery

# this static method prints out informations about
# the dimensionality of the parameters required
Battery.dimensional_info()

# instantiate battery class. Note that the attributes are public
# and can be accessed at any time
# e.g. battery.soc to access the state of charge; or battery.n_cycles 
battery = Battery(soc=0, vn=700, n_cycles=0, 
                  max_kwh=50, pn=250)

# this asks to the battery to charge 10 kw for 1 hour
p_left = battery.charge(10, 60*60)

# this asks to discharge 5 kw for 1 hour
# NB: the 'testing' parameter will print out for
# convenience the estimated value of eta (battery efficiency) and
# return it too
p_left, eta = battery.discharge(5, 60*60, testing=True)