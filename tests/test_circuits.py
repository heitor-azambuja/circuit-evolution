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
