import math

import pytest

import amp_design
import components


def _pot():
    return components.DigitalPot(100_000, 100)


def test_collector_resistor_reproduces_the_requested_gain():
    """The equations must at least be self-consistent, whatever they miss in SPICE."""
    for gain in (5, 10, 15, 20):
        rc = amp_design.collector_resistor(gain)
        result = amp_design.design(gain)
        loaded = rc * amp_design.LOAD / (rc + amp_design.LOAD)
        assert loaded / result['emitter_resistance_small_signal'] == pytest.approx(gain)


def test_higher_gain_needs_a_smaller_collector_resistor():
    resistances = [amp_design.collector_resistor(g) for g in (5, 10, 15, 20)]
    assert resistances == sorted(resistances, reverse=True)


def test_maximum_swing_puts_half_the_supply_across_the_collector_resistor():
    result = amp_design.design(10)
    rc = result['resistances'][2]
    assert rc * result['collector_current'] == pytest.approx(amp_design.VCC / 2)


def test_a_gain_beyond_the_supply_and_load_is_rejected():
    """VCC/(2*VT*Av) must exceed 1, or the criterion asks for a negative Rc."""
    with pytest.raises(ValueError):
        amp_design.collector_resistor(100)


def test_gain_of_five_does_not_fit_a_100k_potentiometer():
    """A measured limitation of the hardware, not of the algorithm."""
    pot = _pot()
    rc = amp_design.collector_resistor(5)
    assert rc > pot.max_value
    assert amp_design.exceeds_pot([rc], pot) == [True]


def test_gains_of_ten_and_above_do_fit():
    pot = _pot()
    for gain in (10, 15, 20):
        assert amp_design.exceeds_pot([amp_design.collector_resistor(gain)], pot) == [False]


def test_textbook_divider_asks_for_more_than_the_pot_can_reach():
    """The classic I_div = 10*I_B rule is unimplementable with this component."""
    pot = _pot()
    result = amp_design.design(10)
    total = amp_design.textbook_divider_total(result['collector_current'], beta=294.3)

    assert total > 10 * (2 * pot.max_value)  # two pots in series is the most available


def test_divider_sets_the_base_voltage_and_fits_the_pot():
    pot = _pot()
    result = amp_design.design(10)
    r1, r2 = amp_design.divider(result['base_voltage'], pot)

    assert r2 / (r1 + r2) * amp_design.VCC == pytest.approx(result['base_voltage'])
    assert max(r1, r2) <= pot.max_value


def test_design_returns_resistances_in_the_circuits_order():
    """BJTClassAAmp.configure_resistors expects R1, R2, Rc, Re."""
    pot = _pot()
    result = amp_design.design(10, divider_total=139_000)
    r1, r2, rc, re = result['resistances']

    assert r1 > r2                      # upper leg is larger for VB < VCC/2
    assert rc == pytest.approx(amp_design.collector_resistor(10))
    assert re == pytest.approx(amp_design.VE_FRACTION * amp_design.VCC
                               / result['collector_current'])
    assert all(math.isfinite(v) for v in result['resistances'])


def test_emitter_resistor_lands_on_coarse_taps_at_high_gain():
    """Why quantization bites hardest where the gain is highest."""
    pot = _pot()
    taps = [amp_design.quantize([amp_design.design(g)['resistances'][3]], pot)[0]
            for g in (10, 15, 20)]
    assert taps == sorted(taps, reverse=True)
    assert taps[-1] < 10                # gain 20 sits in the pot's coarsest region


def test_quantize_saturates_at_both_ends():
    pot = _pot()
    assert amp_design.quantize([-1.0, 1e9], pot) == [1, pot.tap_points]
