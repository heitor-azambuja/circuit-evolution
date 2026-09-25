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


def test_sallen_key_cascades_two_stages_through_an_ideal_buffer():
    netlist = str(circuits.SallenKeyLowPass().circuit)
    assert 'Eopa' in netlist and 'Eopb' in netlist
    assert 'C1b midb out' in netlist  # second stage bootstraps to the final output
