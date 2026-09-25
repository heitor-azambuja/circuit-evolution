import math
import re

import pytest

import circuits


def test_4r_configure_resistors_wrong_length_raises():
    amp = circuits.BJTClassAAmp()
    with pytest.raises(ValueError):
        amp.configure_resistors([1, 2, 3])


def test_8r_configure_resistors_wrong_length_raises():
    amp = circuits.BJTClassAAmp8R()
    with pytest.raises(ValueError):
        amp.configure_resistors([1] * 7)


def test_configure_resistors_clamps_non_positive_values():
    amp = circuits.BJTClassAAmp()
    amp.configure_resistors([0, -5, 100, 100])
    assert float(amp.circuit.R1.resistance) == circuits.BJTClassAAmp._MIN_SPICE_R
    assert float(amp.circuit.R2.resistance) == circuits.BJTClassAAmp._MIN_SPICE_R


def test_configure_resistors_keeps_positive_values():
    amp = circuits.BJTClassAAmp()
    amp.configure_resistors([100, 200, 300, 400])
    assert float(amp.circuit.R1.resistance) == 100
    assert float(amp.circuit.Rc.resistance) == 300


def test_transient_analysis_requires_resistors_configured():
    amp = circuits.BJTClassAAmp()
    with pytest.raises(RuntimeError):
        amp.transient_analysis()


def test_8r_transient_analysis_requires_resistors_configured():
    amp = circuits.BJTClassAAmp8R()
    with pytest.raises(RuntimeError):
        amp.transient_analysis()


def test_sallen_key_configure_resistors_wrong_length_raises():
    filt = circuits.SallenKeyLowPass()
    with pytest.raises(ValueError):
        filt.configure_resistors([1, 2, 3])


def test_sallen_key_configure_resistors_clamps_non_positive_values():
    filt = circuits.SallenKeyLowPass()
    filt.configure_resistors([0, -5, 100, 100])
    assert float(filt.circuit.R1a.resistance) == circuits.SallenKeyLowPass._MIN_SPICE_R
    assert float(filt.circuit.R2a.resistance) == circuits.SallenKeyLowPass._MIN_SPICE_R


def test_sallen_key_configure_resistors_keeps_positive_values():
    filt = circuits.SallenKeyLowPass()
    filt.configure_resistors([100, 200, 300, 400])
    assert float(filt.circuit.R1a.resistance) == 100
    assert float(filt.circuit.R2b.resistance) == 400


def test_sallen_key_ac_analysis_requires_resistors_configured():
    filt = circuits.SallenKeyLowPass()
    filt.configure_capacitors(27, 22, 82, 10)
    with pytest.raises(RuntimeError):
        filt.ac_analysis()


def test_sallen_key_ac_analysis_requires_capacitors_configured():
    filt = circuits.SallenKeyLowPass()
    filt.configure_resistors([1000] * 4)
    with pytest.raises(RuntimeError):
        filt.ac_analysis()


def test_sallen_key_cascades_two_stages():
    netlist = str(circuits.SallenKeyLowPass().circuit)
    assert 'C1b midb out' in netlist  # second stage bootstraps to the final output
    assert 'R1a in mida' in netlist


def test_sallen_key_defaults_to_the_real_part():
    """Experiments run with the MCP6002; the ideal buffer is opt-in."""
    filt = circuits.SallenKeyLowPass()
    netlist = str(filt.circuit)

    assert filt.op_amp == circuits.MCP6002
    assert 'mcp6002.spice' in netlist
    # Both stages instantiate the part, wired as unity-gain buffers: the
    # inverting input and the output are the same node.
    assert 'Xopa opina stagea stagea MCP6002' in netlist
    assert 'Xopb opinb out out MCP6002' in netlist


def test_sallen_key_ideal_op_amp_is_a_plain_unity_buffer():
    netlist = str(circuits.SallenKeyLowPass(op_amp=None).circuit)
    assert 'Eopa' in netlist and 'Eopb' in netlist
    assert 'MCP6002' not in netlist


def test_mcp6002_model_matches_the_datasheet_typicals():
    """Guards the two numbers the filter's accuracy actually hinges on."""
    model = open('mcp6002.spice').read()
    open_loop_gain = float(re.search(r'^Eol .*? ([\d.]+)\s*$', model,
                                     re.MULTILINE).group(1))
    resistance = float(re.search(r'^Rdom .*?([\d.]+)K\s*$', model,
                                 re.MULTILINE).group(1)) * 1e3
    capacitance = float(re.search(r'^Cdom .*?([\d.]+)U\s*$', model,
                                  re.MULTILINE).group(1)) * 1e-6

    # DC Open-Loop Gain, 112 dB typical
    assert 20 * math.log10(open_loop_gain) == pytest.approx(112.0, abs=0.01)
    # Gain Bandwidth Product, 1.0 MHz typical, set by the dominant pole
    pole_hz = 1.0 / (2 * math.pi * resistance * capacitance)
    assert pole_hz * open_loop_gain == pytest.approx(1e6, rel=1e-4)
