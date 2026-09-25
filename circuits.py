import PySpice.Unit as unit
from PySpice.Spice.Netlist import Circuit
from PySpice.Spice.Library import SpiceLibrary

import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging(logging_level='ERROR')

class BJTClassAAmp:
	_MIN_SPICE_R = 1e-3  # NGSpice can't handle true 0 Ω

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
		
		circuit.C('i', 'in', 3)
		circuit.C('e', 5, circuit.gnd)
		circuit.C('o', 4, 'out')

		spice_library = SpiceLibrary('.')
		circuit.include(spice_library['bc547b'])
		circuit.BJT(1, 4, 3, 5, model='bc547b')  # (name, collector, base, emmiter, model)

		self.circuit = circuit
		self._resistors_configured = False


	def configure_resistors(self, values) -> None:
		'''
			Configure circuit Resistors resistance in Ohms.
			The order is: R1, R2, Rc, Re
		'''
		if len(values) != 4:
			raise ValueError('4 resistors values are required!')
		_r = [max(float(v), self._MIN_SPICE_R) for v in values]
		self.circuit.R1.resistance = _r[0]@unit.u_Ohm
		self.circuit.R2.resistance = _r[1]@unit.u_Ohm
		self.circuit.Rc.resistance = _r[2]@unit.u_Ohm
		self.circuit.Re.resistance = _r[3]@unit.u_Ohm
		self._resistors_configured = True


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
		self.circuit.Vs.frequency = freq@unit.u_Hz

	
	def transient_analysis(self, step_time=0.00001, end_time=0.002, temperature=25) -> object:
		if not self._resistors_configured:
			raise RuntimeError('configure_resistors() must be called before transient_analysis()')
		simulator = self.circuit.simulator(temperature=temperature, nominal_temperature=25)
		return simulator.transient(step_time=step_time, end_time=end_time)
	

class SallenKeyLowPass:
	'''
		Cascade of unity-gain Sallen-Key low-pass stages (2 stages => 4th order).

		Per stage:  in --[R1]-- a --[R2]-- b --> unity buffer --> stage output,
		with C1 bootstrapped from a back to the stage output and C2 from b to
		ground. The op-amp is an ideal VCVS of gain 1, so the only departure from
		the textbook response the GA has to fight is the resistor quantization.

		Capacitors are fixed hardware here and must be deliberately unequal: a
		unity-gain stage can only reach Q = 0.5*sqrt(C1/C2), so equal capacitors
		cap Q at 0.5 and put any 4th-order response out of reach. See
		filter_design.max_q.
	'''

	_MIN_SPICE_R = 1e-3  # NGSpice can't handle true 0 Ω
	_STAGE_TAGS = ('a', 'b')

	def __init__(self, sin_ampl=1.0, sin_freq=1000, ckt_name='Sallen-Key Low-Pass') -> None:
		circuit = Circuit(ckt_name)

		circuit.SinusoidalVoltageSource('s', 'in', circuit.gnd,
										amplitude=sin_ampl@unit.u_V,
										frequency=sin_freq@unit.u_Hz,
										ac_magnitude=1@unit.u_V)

		node_in = 'in'
		for index, tag in enumerate(self._STAGE_TAGS):
			mid = f'mid{tag}'
			opamp_in = f'opin{tag}'
			node_out = 'out' if index == len(self._STAGE_TAGS) - 1 else f'stage{tag}'

			circuit.R(f'1{tag}', node_in, mid)
			circuit.R(f'2{tag}', mid, opamp_in)
			circuit.C(f'1{tag}', mid, node_out)
			circuit.C(f'2{tag}', opamp_in, circuit.gnd)
			circuit.VCVS(f'op{tag}', node_out, circuit.gnd, opamp_in, circuit.gnd,
						 voltage_gain=1)

			node_in = node_out

		self.circuit = circuit
		self._resistors_configured = False
		self._capacitors_configured = False


	def configure_resistors(self, values) -> None:
		'''
			Configure circuit Resistors resistance in Ohms.
			The order is: R1a, R2a (first stage), R1b, R2b (second stage)
		'''
		if len(values) != 4:
			raise ValueError('4 resistors values are required!')
		_r = [max(float(v), self._MIN_SPICE_R) for v in values]
		self.circuit.R1a.resistance = _r[0]@unit.u_Ohm
		self.circuit.R2a.resistance = _r[1]@unit.u_Ohm
		self.circuit.R1b.resistance = _r[2]@unit.u_Ohm
		self.circuit.R2b.resistance = _r[3]@unit.u_Ohm
		self._resistors_configured = True


	def configure_capacitors(self, c1a, c2a, c1b, c2b) -> None:
		'''
			Configure circuit Capacitors capacitance in nF.
			Per stage C1 is the bootstrapped one and C2 goes to ground; C1 > C2
			is what buys Q above 0.5.
		'''
		self.circuit.C1a.capacitance = c1a@unit.u_nF
		self.circuit.C2a.capacitance = c2a@unit.u_nF
		self.circuit.C1b.capacitance = c1b@unit.u_nF
		self.circuit.C2b.capacitance = c2b@unit.u_nF
		self._capacitors_configured = True


	def ac_analysis(self, start_frequency=10, stop_frequency=100000,
					points_per_decade=20, temperature=25) -> object:
		if not self._resistors_configured:
			raise RuntimeError('configure_resistors() must be called before ac_analysis()')
		if not self._capacitors_configured:
			raise RuntimeError('configure_capacitors() must be called before ac_analysis()')
		simulator = self.circuit.simulator(temperature=temperature, nominal_temperature=25)
		return simulator.ac(variation='dec', number_of_points=points_per_decade,
							start_frequency=start_frequency@unit.u_Hz,
							stop_frequency=stop_frequency@unit.u_Hz)


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
		self._resistors_configured = False


	def configure_resistors(self, values) -> None:
		'''
			Configure circuit Resistors resistance in Ohms
			The order is: R11, R12, R21, R22, Rc1, Rc2, Re1, Re2
		'''
		
		if len(values) != 8:
			raise ValueError('8 resistors values are required!')
		_r = [max(float(v), self._MIN_SPICE_R) for v in values]
		self.circuit.R11.resistance = _r[0]@unit.u_Ohm
		self.circuit.R12.resistance = _r[1]@unit.u_Ohm
		self.circuit.R21.resistance = _r[2]@unit.u_Ohm
		self.circuit.R22.resistance = _r[3]@unit.u_Ohm
		self.circuit.Rc1.resistance = _r[4]@unit.u_Ohm
		self.circuit.Rc2.resistance = _r[5]@unit.u_Ohm
		self.circuit.Re1.resistance = _r[6]@unit.u_Ohm
		self.circuit.Re2.resistance = _r[7]@unit.u_Ohm
		self._resistors_configured = True