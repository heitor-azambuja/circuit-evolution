import numpy as np
import PySpice.Unit as unit
import matplotlib.pyplot as plt
from PySpice.Spice.Library import SpiceLibrary
from PySpice.Spice.Netlist import Circuit

import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()

circuit = Circuit('test')
		
circuit.V('cc', 1, circuit.gnd, 3.3@unit.u_V)
# circuit.SinusoidalVoltageSource('s')

circuit.R(1, 1, 3, 1000@unit.u_Ohm)
circuit.R(2, 3, circuit.gnd)
circuit.R('c', 1, 4)
# circuit.R('e')
print(circuit)