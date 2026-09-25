import numpy as np
import pytest
import scipy.signal

import components
import filter_design


def test_butterworth_4th_order_stage_qs_match_textbook_values():
    stages = filter_design.stage_parameters(4, filter_design.BUTTERWORTH, 1000.0)
    qs = [q for _, q in stages]
    assert len(stages) == 2
    assert qs[0] == pytest.approx(0.5412, abs=1e-4)
    assert qs[1] == pytest.approx(1.3065, abs=1e-4)


def test_butterworth_stages_share_the_cutoff_frequency():
    stages = filter_design.stage_parameters(4, filter_design.BUTTERWORTH, 1000.0)
    for wo, _ in stages:
        assert wo == pytest.approx(2 * np.pi * 1000.0, rel=1e-9)


def test_stages_are_ordered_by_ascending_q():
    stages = filter_design.stage_parameters(6, filter_design.BUTTERWORTH, 1000.0)
    qs = [q for _, q in stages]
    assert len(stages) == 3
    assert qs == sorted(qs)


def test_chebyshev_needs_higher_q_than_butterworth():
    butter = filter_design.stage_parameters(4, filter_design.BUTTERWORTH, 1000.0)
    cheby = filter_design.stage_parameters(4, filter_design.CHEBYSHEV, 1000.0, ripple_db=0.5)
    assert cheby[-1][1] > butter[-1][1]


def test_odd_order_is_rejected():
    with pytest.raises(ValueError):
        filter_design.stage_parameters(3, filter_design.BUTTERWORTH, 1000.0)


def test_chebyshev_without_ripple_is_rejected():
    with pytest.raises(ValueError):
        filter_design.stage_parameters(4, filter_design.CHEBYSHEV, 1000.0)


def test_unknown_response_is_rejected():
    with pytest.raises(ValueError):
        filter_design.stage_parameters(4, 'bessel', 1000.0)


def test_equal_capacitors_cap_q_at_one_half():
    """The constraint that forces unequal capacitors in the Sallen-Key design."""
    assert filter_design.max_q(10e-9, 10e-9) == pytest.approx(0.5)


def test_max_q_grows_with_the_square_root_of_the_capacitor_ratio():
    assert filter_design.max_q(100e-9, 10e-9) == pytest.approx(0.5 * np.sqrt(10))


def test_butterworth_high_q_stage_is_unreachable_with_equal_capacitors():
    _, q = filter_design.stage_parameters(4, filter_design.BUTTERWORTH, 1000.0)[-1]
    with pytest.raises(ValueError):
        filter_design.ideal_resistors(2 * np.pi * 1000.0, q, 10e-9, 10e-9)


def test_ideal_resistors_reproduce_the_requested_wo_and_q():
    c1, c2 = 100e-9, 10e-9
    wo, q = 2 * np.pi * 1000.0, 1.3065
    r1, r2 = filter_design.ideal_resistors(wo, q, c1, c2)

    assert 1.0 / np.sqrt(r1 * r2 * c1 * c2) == pytest.approx(wo, rel=1e-9)
    assert np.sqrt(r1 * r2 * c1 / c2) / (r1 + r2) == pytest.approx(q, rel=1e-9)


def test_ideal_resistors_returns_larger_value_first():
    r1, r2 = filter_design.ideal_resistors(2 * np.pi * 1000.0, 1.0, 100e-9, 10e-9)
    assert r1 >= r2 > 0


def test_q_at_the_ceiling_gives_equal_resistors():
    c1, c2 = 100e-9, 10e-9
    r1, r2 = filter_design.ideal_resistors(2 * np.pi * 1000.0, filter_design.max_q(c1, c2), c1, c2)
    assert r1 == pytest.approx(r2)


def test_transfer_function_matches_the_requested_response():
    """The strongest check: designed resistors must reproduce scipy's own curve."""
    fc, c2 = 1000.0, 10e-9
    c1 = 100e-9
    freqs = np.logspace(1, 5, 81)
    w = 2 * np.pi * freqs

    num, den = [1.0], [1.0]
    for wo, q in filter_design.stage_parameters(4, filter_design.BUTTERWORTH, fc):
        r1, r2 = filter_design.ideal_resistors(wo, q, c1, c2)
        stage_num, stage_den = filter_design.transfer_function(r1, r2, c1, c2)
        num = np.polymul(num, stage_num)
        den = np.polymul(den, stage_den)

    _, cascade = scipy.signal.freqs(num, den, worN=w)
    b, a = scipy.signal.butter(4, 2 * np.pi * fc, btype='low', analog=True)
    _, reference = scipy.signal.freqs(b, a, worN=w)

    cascade_db = 20 * np.log10(np.abs(cascade))
    reference_db = 20 * np.log10(np.abs(reference))
    assert np.max(np.abs(cascade_db - reference_db)) < 1e-6


def test_swapping_a_stages_resistors_leaves_the_response_unchanged():
    c1, c2 = 100e-9, 10e-9
    r1, r2 = filter_design.ideal_resistors(2 * np.pi * 1000.0, 1.2, c1, c2)
    assert filter_design.transfer_function(r1, r2, c1, c2) == \
        filter_design.transfer_function(r2, r1, c1, c2)


def test_quantize_to_pot_rounds_to_the_nearest_tap():
    pot = components.DigitalPot(10000, 100)
    step = pot.min_value
    taps = filter_design.quantize_to_pot([0.0, step, 2 * step, 2.4 * step, 2.6 * step], pot)
    assert taps == [1, 2, 3, 3, 4]


def test_quantize_to_pot_round_trips_through_get_resistance():
    pot = components.DigitalPot(10000, 100)
    exact = pot.get_resistance(37)
    assert filter_design.quantize_to_pot([exact], pot) == [37]


def test_quantize_to_pot_saturates_at_both_ends():
    pot = components.DigitalPot(10000, 100)
    assert filter_design.quantize_to_pot([-500.0, 1e9], pot) == [1, pot.tap_points]
