import PySpice.Unit as unit
from PySpice.Spice.Netlist import Circuit
from PySpice.Spice.Library import SpiceLibrary

import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()


class BJTClassAAmp:
	def __init__(self, sin_dc_offset=0, sin_ampl=0.01, sin_freq=1000, vcc=3.3, ckt_name='BJT Class 1 Amplifier', load=10000) -> None:
		circuit = Circuit(ckt_name)
		
		circuit.V('cc', 1, circuit.gnd, vcc@unit.u_V)
		circuit.SinusoidalVoltageSource('s', 'in', circuit.gnd, 
								  		dc_offset=sin_dc_offset@unit.u_V,
										amplitude=sin_ampl@unit.u_V, 
										frequency=sin_freq@unit.u_Hz)

		circuit.R(1, 1, 3)
		circuit.R(2, 3, circuit.gnd)
		circuit.R('c', 1, 4)
		circuit.R('e', 5, circuit.gnd)
		circuit.R('l', 'out', circuit.gnd, load@unit.u_Ohm)
		
		# circuit.C('i', 3, 'in')
		circuit.C('i', 'in', 3)
		circuit.C('e', 5, circuit.gnd)
		circuit.C('o', 4, 'out')

		spice_library = SpiceLibrary('.')
		circuit.include(spice_library['bc547b'])
		circuit.BJT(1, 4, 3, 5, model='bc547b')  # (name, collector, base, emmiter, model)

		self.circuit = circuit


	def configure_resistors(self, values) -> None:
		'''
			Configure circuit Resistors resistance in Ohms.
			The order is: R1, R2, Rc, Re
		'''
		if len(values) != 4:
			raise ValueError('4 resistors values are required!')
		
		self.circuit.R1.resistance = values[0]@unit.u_Ohm
		self.circuit.R2.resistance = values[1]@unit.u_Ohm
		self.circuit.Rc.resistance = values[2]@unit.u_Ohm
		self.circuit.Re.resistance = values[3]@unit.u_Ohm


	def configure_capacitors(self, cin, ce, cout) -> None:
		'''
			Configure circuit Capacitors capacitance in uF
		'''
		self.circuit.Ci.capacitance = cin@unit.u_uF
		self.circuit.Ce.capacitance = ce@unit.u_uF
		self.circuit.Co.capacitance = cout@unit.u_uF


	def configure_vcc(self, vcc=3.3) -> None:
		self.circuit.Vcc.dc_value = vcc@unit.u_V


	def configure_input_signal(self, ampl=0.01, dc_offset=0, freq=1000) -> None:
		self.circuit.Vs.amplitude = ampl@unit.u_V
		self.circuit.Vs.dc_offset = dc_offset@unit.u_V
		self.circuit.frequency = freq@unit.u_Hz

	
	def transient_analysis(self, step_time=0.00001, end_time=0.002) -> object:
		simulator = self.circuit.simulator(temperature=25, nominal_temperature=25)
		return simulator.transient(step_time=step_time, end_time=end_time)
	

class BJTClassAAmp8R(BJTClassAAmp):
	def __init__(self, sin_dc_offset=0, sin_ampl=0.01, sin_freq=1000, vcc=3.3, ckt_name='BJT Class 1 Amplifier', load=10000) -> None:
		circuit = Circuit(ckt_name)
		
		circuit.V('cc', 1, circuit.gnd, vcc@unit.u_V)
		circuit.SinusoidalVoltageSource('s', 'in', circuit.gnd, 
								  		dc_offset=sin_dc_offset@unit.u_V,
										amplitude=sin_ampl@unit.u_V, 
										frequency=sin_freq@unit.u_Hz)

		circuit.R(11, 8, 3)
		circuit.R(12, 1, 8)
		circuit.R(21, 3, 7)
		circuit.R(22, 7, circuit.gnd)
		circuit.R('c1', 9, 4)
		circuit.R('c2', 1, 9)
		circuit.R('e1', 5, 6)
		circuit.R('e2', 6, circuit.gnd)
		
		circuit.R('l', 'out', circuit.gnd, load@unit.u_Ohm)
		
		circuit.C('i', 'in', 3)
		circuit.C('e', 5, circuit.gnd)
		circuit.C('o', 4, 'out')

		spice_library = SpiceLibrary('.')
		circuit.include(spice_library['bc547b'])
		circuit.BJT(1, 4, 3, 5, model='bc547b')  # (name, collector, base, emmiter, model)

		self.circuit = circuit


	def configure_resistors(self, values) -> None:
		'''
			Configure circuit Resistors resistance in Ohms
			The order is: R11, R12, R21, R22, Rc1, Rc2, Re1, Re2
		'''
		
		if len(values) != 8:
			raise ValueError('8 resistors values are required!')
		
		self.circuit.R11.resistance = values[0]@unit.u_Ohm
		self.circuit.R12.resistance = values[1]@unit.u_Ohm
		self.circuit.R21.resistance = values[2]@unit.u_Ohm
		self.circuit.R22.resistance = values[3]@unit.u_Ohm
		self.circuit.Rc1.resistance = values[4]@unit.u_Ohm
		self.circuit.Rc2.resistance = values[5]@unit.u_Ohm
		self.circuit.Re1.resistance = values[6]@unit.u_Ohm
		self.circuit.Re2.resistance = values[7]@unit.u_Ohm


# class SallenKeyFilter: