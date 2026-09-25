import numpy as np
import pytest

import components
import filter_sk4_evolution as sk4
import filter_sk8_evolution as sk8
import filter_tolerance


def test_wiper_resistance_adds_in_series_at_every_tap():
    """X9C datasheet: 40 ohm typical, 100 ohm max, in series in rheostat mode."""
    plain = components.DigitalPot(10000, 100)
    with_wiper = components.DigitalPot(10000, 100, wiper_resistance=40.0)

    for tap in (1, 50, 100):
        assert with_wiper.get_resistance(tap) == pytest.approx(
            plain.get_resistance(tap) + 40.0)


def test_wiper_resistance_defaults_to_zero_so_earlier_results_stay_comparable():
    assert components.DigitalPot(10000, 100).wiper_resistance == 0.0


def test_the_wiper_is_several_tap_steps_on_the_fine_part():
    """Why it cannot be dismissed as rounding: 40 ohm on a 1k pot is four steps."""
    fine = components.DigitalPot(1000, 100)
    assert filter_tolerance.WIPER_RESISTANCE_TYP / fine.min_value > 3


def test_a_board_is_reproducible_from_its_index():
    assert filter_tolerance.draw_board(7, 4) == filter_tolerance.draw_board(7, 4)


def test_different_boards_differ():
    assert filter_tolerance.draw_board(1, 4) != filter_tolerance.draw_board(2, 4)


def test_deviations_stay_inside_the_datasheet_limits():
    for index in range(1, 40):
        board = filter_tolerance.draw_board(index, 8)
        assert all(1 - filter_tolerance.POT_TOLERANCE <= s <= 1 + filter_tolerance.POT_TOLERANCE
                   for s in board.pot_scale)
        assert all(1 - filter_tolerance.CAPACITOR_TOLERANCE <= s
                   <= 1 + filter_tolerance.CAPACITOR_TOLERANCE
                   for s in board.capacitor_scale)
        assert all(0 <= w <= filter_tolerance.WIPER_RESISTANCE_MAX for w in board.pot_wiper)


def test_one_deviation_is_drawn_per_potentiometer_on_the_board():
    """Each resistor is a separate part, so each gets its own draw."""
    assert len(filter_tolerance.draw_board(1, 4).pot_scale) == 4
    assert len(filter_tolerance.draw_board(1, 8).pot_scale) == 8
    assert len(sk4.POT_SPECS) == 4
    assert len(sk8.POT_SPECS) == 8


def test_a_board_shifts_the_resistance_the_same_taps_produce():
    board = filter_tolerance.draw_board(1, len(sk4.POT_SPECS))
    spec = filter_tolerance.board_spec(sk4, board)

    nominal = sk4.resistor_mapper([50] * 4)
    actual = spec.resistor_mapper([50] * 4)

    assert nominal == [nominal[0]] * 4          # identical parts, nominally
    assert len(set(actual)) == 4                # and all different on a real board
    for a, b in zip(nominal, actual):
        assert a != pytest.approx(b)


def test_a_board_shifts_the_capacitors_too():
    board = filter_tolerance.draw_board(3, len(sk8.POT_SPECS))
    spec = filter_tolerance.board_spec(sk8, board)

    class _Recorder:
        def configure_capacitors(self, *values):
            self.values = values

    recorder = _Recorder()
    spec.setup_hook(recorder, 'butterworth_1000')

    nominal = (27.0, 22.0, 82.0, 10.0)
    assert recorder.values != nominal
    for got, want, scale in zip(recorder.values, nominal, board.capacitor_scale):
        assert got == pytest.approx(want * scale)


def test_the_board_spec_keeps_everything_else_from_the_module():
    board = filter_tolerance.draw_board(2, len(sk4.POT_SPECS))
    spec = filter_tolerance.board_spec(sk4, board)

    assert spec.circuit_name == sk4.SPEC.circuit_name
    assert spec.num_genes == sk4.SPEC.num_genes
    assert spec.evaluator is sk4.SPEC.evaluator
