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


	def configure_resistors(self, r1, r2, rc, re) -> None:
		'''
			Configure circuit Resistors resistance in Ohms
		'''
		self.circuit.R1.resistance = r1@unit.u_Ohm
		self.circuit.R2.resistance = r2@unit.u_Ohm
		self.circuit.Rc.resistance = rc@unit.u_Ohm
		self.circuit.Re.resistance = re@unit.u_Ohm


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


	def configure_resistors(self, r11, r12, r21, r22, rc1, rc2, re1, re2) -> None:
		'''
			Configure circuit Resistors resistance in Ohms
		'''
		self.circuit.R11.resistance = r11@unit.u_Ohm
		self.circuit.R12.resistance = r12@unit.u_Ohm
		self.circuit.R21.resistance = r21@unit.u_Ohm
		self.circuit.R22.resistance = r22@unit.u_Ohm
		self.circuit.Rc1.resistance = rc1@unit.u_Ohm
		self.circuit.Rc2.resistance = rc2@unit.u_Ohm
		self.circuit.Re1.resistance = re1@unit.u_Ohm
		self.circuit.Re2.resistance = re2@unit.u_Ohm


# class SallenKeyFilter: