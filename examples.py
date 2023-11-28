import numpy as np
import PySpice.Unit as unit
import matplotlib.pyplot as plt
from PySpice.Spice.Library import SpiceLibrary
from PySpice.Spice.Netlist import Circuit

import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()


def plot_npn_bjt_curves(component: str) -> None:
    '''
    Example from the docs
    '''
    spice_library = SpiceLibrary('.')

    figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 10))
    figure.suptitle(f'{component} Characteristc Curves')

    circuit = Circuit('Transistor')

    Vbase = circuit.V('base', '1', circuit.gnd, 1@unit.u_V)
    circuit.R('base', 1, 'base', 1@unit.u_kΩ)
    Vcollector = circuit.V('collector', '2', circuit.gnd, 0@unit.u_V)
    circuit.R('collector', 2, 'collector', 1@unit.u_kΩ)
    # circuit.BJT(1, 'collector', 'base', circuit.gnd, model='generic')
    # circuit.model('generic', 'npn')
    circuit.include(spice_library[component])
    # circuit.include(spice_library['BC547B'])
    circuit.BJT(1, 'collector', 'base', circuit.gnd, model=component)

    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.dc(Vbase=slice(0, 3, .01))

    ax1.plot(analysis.base, unit.u_mA(-analysis.Vbase)) # Fixme: I_Vbase
    ax1.axvline(x=.65, color='red')
    ax1.legend(('Base-Emitter Diode curve',), loc=(.1,.8))
    ax1.grid()
    ax1.set_xlabel('Vbe [V]')
    ax1.set_ylabel('Ib [mA]')

    circuit = Circuit('Transistor')
    Ibase = circuit.I('base', circuit.gnd, 'base', 10@unit.u_uA) # take care to the orientation
    Vcollector = circuit.V('collector', 'collector', circuit.gnd, 5)
    # circuit.BJT(1, 'collector', 'base', circuit.gnd, model='generic')
    # circuit.model('generic', 'npn')
    circuit.include(spice_library[component])
    circuit.BJT(1, 'collector', 'base', circuit.gnd, model=component)

    ax2.grid()
    # ax2.legend(('Ic(Vce, Ib)',), loc=(.5,.5))
    ax2.set_xlabel('Vce [V]')
    ax2.set_ylabel('Ic [mA]')
    ax2.axvline(x=.2, color='red')

    ax3.grid()
    # ax3.legend(('beta(Vce)',), loc=(.5,.5))
    ax3.set_xlabel('Vce [V]')
    ax3.set_ylabel('beta')
    ax3.axvline(x=.2, color='red')

    for base_current in np.arange(0, 100, 10):
        base_current = base_current@unit.u_uA
        Ibase.dc_value = base_current
        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.dc(Vcollector=slice(0, 5, .01))
        # add ib as text, linear and saturate region
        # Plot Ic = f(Vce)
        ax2.plot(analysis.collector, unit.u_mA(-analysis.Vcollector))
        # Plot β = Ic / Ib = f(Vce)
        ax3.plot(analysis.collector, -analysis.Vcollector/float(base_current))
        # trans-resistance U = RI   R = U / I = Vce / Ie
        # ax3.plot(analysis.collector, analysis.sweep/(float(base_current)-analysis.Vcollector))
        # Fixme: sweep is not so explicit

    ax4.grid()
    ax4.set_xlabel('Ib [uA]')
    ax4.set_ylabel('Ic [mA]')

    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.dc(Ibase=slice(0, 100e-6, 10e-6))
    # Fixme: sweep
    ax4.plot(analysis.sweep*1e6, unit.u_mA(-analysis.Vcollector), 'o-')
    ax4.legend(('Ic(Ib)',), loc=(.1,.8))

    plt.tight_layout()
    plt.show()


plot_npn_bjt_curves('2n2222a')
