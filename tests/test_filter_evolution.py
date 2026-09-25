import numpy as np
import pytest
import scipy.signal

import circuits
import evolution_common
import filter_design
import filter_evaluation

FC = 1000.0
TARGETS = {
    'butterworth': {'response': filter_design.BUTTERWORTH, 'ripple_db': None},
    'chebyshev': {'response': filter_design.CHEBYSHEV, 'ripple_db': 0.5},
}


def _evaluator():
    return filter_evaluation.AcResponseEvaluator(fc_hz=FC, order=4, targets=TARGETS)


def _sweep_freqs(start=10, stop=100_000, points_per_decade=20):
    decades = np.log10(stop / start)
    return start * 10 ** np.linspace(0, decades, int(decades * points_per_decade) + 1)


class _CannedAcAnalysis:
    def __init__(self, freqs, response):
        self.frequency = freqs
        self._nodes = {'out': response}

    def __getitem__(self, key):
        return self._nodes[key]


class _FakeFilter:
    """Stub filter whose response is a chosen analog transfer function, no SPICE."""

    def __init__(self, num, den, freqs=None):
        # A cascade of unity-gain stages has gain 1 at DC, whatever its poles, so
        # the stub is normalised the same way the real circuit is constrained.
        self.num = np.asarray(num, dtype=float) / (num[-1] / den[-1])
        self.den = np.asarray(den, dtype=float)
        self.freqs = _sweep_freqs() if freqs is None else freqs
        self.resistances = None
        self.capacitors = None

    def configure_resistors(self, values):
        self.resistances = list(values)

    def configure_capacitors(self, *args):
        self.capacitors = args

    def ac_analysis(self, **kwargs):
        _, h = scipy.signal.freqs(self.num, self.den, worN=2 * np.pi * self.freqs)
        return _CannedAcAnalysis(self.freqs, h)


def _exact_filter(target):
    spec = TARGETS[target]
    if spec['response'] == filter_design.BUTTERWORTH:
        b, a = scipy.signal.butter(4, 2 * np.pi * FC, btype='low', analog=True)
    else:
        b, a = scipy.signal.cheby1(4, spec['ripple_db'], 2 * np.pi * FC,
                                   btype='low', analog=True)
    return _FakeFilter(b, a)


def _detuned_filter(target, factor):
    """The same response with its cutoff shifted, i.e. a deliberately worse circuit."""
    spec = TARGETS[target]
    if spec['response'] == filter_design.BUTTERWORTH:
        b, a = scipy.signal.butter(4, 2 * np.pi * FC * factor, btype='low', analog=True)
    else:
        b, a = scipy.signal.cheby1(4, spec['ripple_db'], 2 * np.pi * FC * factor,
                                   btype='low', analog=True)
    return _FakeFilter(b, a)


def test_target_curve_matches_scipy_independently():
    freqs = _sweep_freqs()
    b, a = scipy.signal.butter(4, 2 * np.pi * FC, btype='low', analog=True)
    _, h = scipy.signal.freqs(b, a, worN=2 * np.pi * freqs)

    produced = _evaluator().target_db(freqs, 'butterworth')

    assert np.allclose(produced, 20 * np.log10(np.abs(h)))


def test_every_target_is_normalised_to_unity_gain_at_dc():
    """A unity-gain cascade sits at 0 dB at DC; an even-order Chebyshev does not.

    Without this normalisation the circuit is charged a constant offset across the
    whole passband that no choice of resistors could ever remove.
    """
    evaluator = _evaluator()
    near_dc = np.array([1e-3, 1e-2])
    for target in TARGETS:
        assert evaluator.target_db(near_dc, target) == pytest.approx(0.0, abs=1e-6)


def test_normalisation_shifts_chebyshev_but_not_butterworth():
    evaluator = _evaluator()
    freqs = _sweep_freqs()

    b, a = scipy.signal.cheby1(4, 0.5, 2 * np.pi * FC, btype='low', analog=True)
    _, h = scipy.signal.freqs(b, a, worN=2 * np.pi * freqs)
    raw_db = 20 * np.log10(np.abs(h))

    shift = evaluator.target_db(freqs, 'chebyshev') - raw_db
    assert shift == pytest.approx(0.5, abs=1e-6)


def test_chebyshev_target_differs_from_butterworth():
    freqs = _sweep_freqs()
    evaluator = _evaluator()
    assert not np.allclose(evaluator.target_db(freqs, 'butterworth'),
                           evaluator.target_db(freqs, 'chebyshev'))


def test_unknown_response_in_targets_is_rejected():
    with pytest.raises(ValueError):
        filter_evaluation.AcResponseEvaluator(
            fc_hz=FC, order=4, targets={'bessel': {'response': 'bessel'}})


def test_exact_response_scores_the_maximum_fitness():
    for target in TARGETS:
        fitness = _evaluator().fitness(_exact_filter(target), [1e3] * 4, target)
        assert fitness == pytest.approx(1.0)


def test_fitness_is_bounded_above_by_one():
    fitness = _evaluator().fitness(_detuned_filter('butterworth', 3.0), [1e3] * 4,
                                  'butterworth')
    assert 0.0 < fitness < 1.0


def test_fitness_decreases_monotonically_as_the_response_is_detuned():
    evaluator = _evaluator()
    fitnesses = [
        evaluator.fitness(_detuned_filter('butterworth', factor), [1e3] * 4, 'butterworth')
        for factor in (1.0, 1.1, 1.5, 2.0, 4.0)
    ]
    assert fitnesses == sorted(fitnesses, reverse=True)


def test_passband_errors_are_weighted_above_stopband_errors():
    """A given dB error costs more in the passband than in the stopband."""
    evaluator = _evaluator()
    freqs = _sweep_freqs()
    reference = evaluator.target_db(freqs, 'butterworth')

    in_passband = freqs <= FC
    in_stopband = freqs > 10 * FC

    passband_hit = reference.copy()
    passband_hit[in_passband] += 1.0
    stopband_hit = reference.copy()
    stopband_hit[in_stopband] += 1.0

    assert evaluator.weighted_rmse_db(freqs, passband_hit, 'butterworth') > \
        evaluator.weighted_rmse_db(freqs, stopband_hit, 'butterworth')


def test_error_below_the_noise_floor_is_ignored():
    """Both curves clipped at the floor means deep-stopband mismatch costs nothing."""
    evaluator = _evaluator()
    freqs = _sweep_freqs()
    reference = evaluator.target_db(freqs, 'butterworth')

    very_deep = reference < -120
    assert very_deep.any(), 'sweep should reach past the floor'
    mangled = reference.copy()
    mangled[very_deep] = -95.0

    assert evaluator.weighted_rmse_db(freqs, mangled, 'butterworth') == pytest.approx(0.0)


def test_metrics_report_the_realized_cutoff_and_resistors():
    metrics = _evaluator().metrics(_exact_filter('butterworth'), [8107.5, 5259.8, 8581.5, 3599.7],
                                   'butterworth')

    assert metrics['cutoff_realized_hz'] == pytest.approx(FC, rel=0.02)
    assert metrics['max_passband_error_db'] == pytest.approx(0.0, abs=1e-9)
    assert metrics['resistances'] == [8107.5, 5259.8, 8581.5, 3599.7]


def test_fourth_order_rolls_off_eighty_db_per_decade():
    metrics = _evaluator().metrics(_exact_filter('butterworth'), [1e3] * 4, 'butterworth')
    assert metrics['attenuation_at_10fc_db'] == pytest.approx(-80.0, abs=0.5)


def test_cutoff_is_reported_as_missing_when_never_reached():
    flat = _FakeFilter([1.0], [1.0])
    metrics = _evaluator().metrics(flat, [1e3] * 4, 'butterworth')
    assert metrics['cutoff_realized_hz'] is None


def test_target_slug_and_fields_identify_the_filter_cell():
    evaluator = _evaluator()
    assert evaluator.target_slug('butterworth') == 'butterworth4p'
    assert evaluator.target_fields('chebyshev') == {
        'response': 'chebyshev', 'order': 4, 'cutoff_hz': FC, 'ripple_db': 0.5}


def test_target_from_row_recovers_the_response():
    evaluator = _evaluator()
    assert evaluator.target_from_row({'response': 'chebyshev'}) == 'chebyshev'


def test_target_from_row_rejects_a_response_this_evaluator_does_not_know():
    evaluator = _evaluator()
    assert evaluator.target_from_row({'response': 'bessel'}) is None
    assert evaluator.target_from_row({}) is None


def test_every_run_script_registers_a_distinct_circuit_name():
    """plot_results builds its registry from these SPECs, so names must not collide."""
    import plot_results

    specs = plot_results._specs()
    assert 'sallen_key_lp_4p' in specs
    assert 'sallen_key_lp_4p_8r' in specs
    assert len(specs) == 4


def test_8r_mapper_sums_a_coarse_and_a_fine_pot_per_resistor():
    import filter_sk8_evolution as sk8

    solution = [50, 20, 30, 80, 90, 10, 60, 40]
    expected = [
        sk8.pot_10k.get_resistance(50) + sk8.pot_1k.get_resistance(20),
        sk8.pot_10k.get_resistance(30) + sk8.pot_1k.get_resistance(80),
        sk8.pot_10k.get_resistance(90) + sk8.pot_1k.get_resistance(10),
        sk8.pot_10k.get_resistance(60) + sk8.pot_1k.get_resistance(40),
    ]
    assert sk8.resistor_mapper(solution) == expected


def test_8r_mapper_feeds_the_circuit_four_resistances_from_eight_genes():
    """The circuit is unchanged — only how each resistance is realized differs."""
    import filter_sk8_evolution as sk8

    assert sk8.SPEC.num_genes == 8
    assert len(sk8.resistor_mapper([50] * 8)) == 4
    assert sk8.SPEC.circuit_factory is circuits.SallenKeyLowPass


def test_both_filter_variants_share_the_same_targets_and_capacitors():
    """Holding the capacitors fixed is what makes 4R vs 8R a resolution comparison."""
    import filter_sk4_evolution as sk4
    import filter_sk8_evolution as sk8

    assert sk4.SPEC.setup_hook is sk8.SPEC.setup_hook
    assert sk4.SPEC.evaluator.targets is sk8.SPEC.evaluator.targets


def test_8r_realizes_the_ideal_resistors_more_closely_than_4r():
    import filter_baseline
    import filter_sk4_evolution as sk4
    import filter_sk8_evolution as sk8

    coarse = filter_baseline.design(sk4, 'chebyshev')
    fine = filter_baseline.design(sk8, 'chebyshev')

    ideal = np.array(coarse['ideal_resistors'])
    error_4r = np.abs(np.array(coarse['quantized_resistors']) - ideal).max()
    error_8r = np.abs(np.array(fine['quantized_resistors']) - ideal).max()

    assert error_8r < error_4r


def test_evolution_run_drives_the_filter_evaluator():
    """The GA plumbing must work with a frequency-domain evaluator unchanged."""
    circuit = _exact_filter('butterworth')
    spec = evolution_common.CircuitSpec(
        circuit_name='sk_fake',
        display_name='Fake Sallen-Key',
        circuit_factory=lambda: circuit,
        num_genes=4,
        resistor_mapper=lambda solution: [float(v) * 100 for v in solution],
        evaluator=_evaluator(),
        setup_hook=lambda ckt, target: ckt.configure_capacitors(27, 22, 82, 10),
        default_population=4,
        default_generations=10,
    )
    run = evolution_common.EvolutionRun(spec, target='butterworth', exec_counter=1, seed=1)

    class _FakeGA:
        generations_completed = 0

    fitness = run.fitness_func(ga_instance=_FakeGA(), solution=[1, 2, 3, 4], solution_idx=0)

    assert fitness == pytest.approx(1.0)
    assert circuit.capacitors == (27, 22, 82, 10)
    assert circuit.resistances == [100.0, 200.0, 300.0, 400.0]
    assert run.data_json['response'] == 'butterworth'
    assert run._eval_failures == 0
