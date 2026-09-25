import numpy as np
import pytest
import scipy.spatial

import components
import evolution_common


class _CannedAnalysis:
    """Stands in for a PySpice transient analysis result."""

    def __init__(self, vin, vout, time):
        self.out = vout
        self.time = time
        self._nodes = {'in': vin}

    def __getitem__(self, key):
        return self._nodes[key]


class _CannedAmp:
    """Stub circuit returning a fixed waveform, so fitness is exactly predictable."""

    def __init__(self):
        self.resistances = None
        self.capacitors = None
        t = np.linspace(0, 0.002, 41)
        self.vin = 0.01 * np.sin(2 * np.pi * 1000 * t)
        self.vout = -9.5 * self.vin + 0.0004      # deliberately off a gain of 10
        self.time = t

    def configure_capacitors(self, *args):
        self.capacitors = args

    def configure_resistors(self, values):
        self.resistances = list(values)

    def transient_analysis(self, *args, **kwargs):
        return _CannedAnalysis(self.vin, self.vout, self.time)


class _FakeGA:
    """Minimal stand-in for the pygad instance fitness_func reads from."""

    def __init__(self, generations_completed=0):
        self.generations_completed = generations_completed


def _canned_spec(circuit):
    return evolution_common.CircuitSpec(
        circuit_name='canned',
        display_name='Canned Amp',
        circuit_factory=lambda: circuit,
        num_genes=4,
        resistor_mapper=lambda solution: [float(v) * 1000 for v in solution],
        default_population=4,
        default_generations=10,
    )


def _expected_amp_fitness(circuit, gain):
    """Recompute the amplifier objective independently of the implementation."""
    output = np.array(circuit.vout)
    desired = -gain * np.array(circuit.vin)
    steady_start = int(0.25 * len(output))
    distance = scipy.spatial.distance.euclidean(output[steady_start:], desired[steady_start:])
    return 1.0 / distance


def test_amplifier_fitness_matches_the_time_domain_objective():
    """Regression pin: the amplifier objective must survive the evaluator refactor."""
    circuit = _CannedAmp()
    run = evolution_common.EvolutionRun(_canned_spec(circuit), target=10, exec_counter=1, seed=1)

    fitness = run.fitness_func(ga_instance=_FakeGA(), solution=[1, 2, 3, 4], solution_idx=0)

    assert fitness == pytest.approx(_expected_amp_fitness(circuit, 10), rel=1e-12)
    assert run._eval_failures == 0


def test_amplifier_run_configures_resistors_through_the_mapper():
    circuit = _CannedAmp()
    run = evolution_common.EvolutionRun(_canned_spec(circuit), target=10, exec_counter=1, seed=1)

    run.fitness_func(ga_instance=_FakeGA(), solution=[1, 2, 3, 4], solution_idx=0)

    assert circuit.resistances == [1000.0, 2000.0, 3000.0, 4000.0]


def test_amplifier_data_json_records_the_gain_under_its_original_key():
    """The CSV/filename contract for existing amplifier data must not drift."""
    run = evolution_common.EvolutionRun(_canned_spec(_CannedAmp()), target=15,
                                        exec_counter=3, seed=7)

    assert run.data_json['desired_gain'] == 15
    assert run.data_json['ckt_name'] == 'canned'
    assert run.data_json['exec_counter'] == 3
    assert run.data_json['seed'] == 7


class _FakeGAForMutation:
    def __init__(self, num_genes=4, low=1, high=100, mutation_probability=1.0):
        self.gene_space = [range(low, high + 1)] * num_genes
        self.mutation_probability = mutation_probability


def _mutate(offspring, **kwargs):
    return evolution_common.tap_mutation(np.array(offspring),
                                         _FakeGAForMutation(**kwargs))


def test_tap_mutation_keeps_every_gene_inside_its_space():
    np.random.seed(0)
    mutated = _mutate([[1, 1, 100, 100]] * 50)
    assert mutated.min() >= 1
    assert mutated.max() <= 100


def test_tap_mutation_preserves_the_offspring_shape():
    mutated = _mutate([[50, 50, 50, 50], [20, 20, 20, 20]])
    assert mutated.shape == (2, 4)


def test_tap_mutation_leaves_genes_alone_when_probability_is_zero():
    original = [[50, 60, 70, 80]]
    assert (_mutate(original, mutation_probability=0.0) == np.array(original)).all()


def test_tap_mutation_mostly_takes_small_local_steps():
    """The point of the operator: it can refine a solution by a tap or two."""
    np.random.seed(1)
    start = 50
    mutated = _mutate([[start] * 4] * 200)
    steps = np.abs(mutated.flatten() - start)

    moved = steps[steps > 0]
    local = (moved <= evolution_common._LOCAL_MUTATION_STEP).mean()
    assert local > 0.6, f'only {local:.0%} of mutations were local'


def test_tap_mutation_still_makes_distant_jumps():
    """Exploration must survive, otherwise the GA cannot leave a local basin."""
    np.random.seed(2)
    mutated = _mutate([[50] * 4] * 200)
    steps = np.abs(mutated.flatten() - 50)
    assert (steps > evolution_common._LOCAL_MUTATION_STEP).any()


def test_default_spec_uses_the_local_tap_mutation():
    assert _canned_spec(_CannedAmp()).mutation_type is evolution_common.tap_mutation


def test_amplifier_target_from_row_keeps_whole_gains_integral():
    """Regenerated filenames must match the originals, i.e. gain10 and not gain10.0."""
    evaluator = evolution_common.TransientGainEvaluator()
    target = evaluator.target_from_row({'desired_gain': '10.0'})
    assert target == 10
    assert evaluator.target_slug(target) == 'gain10'


def test_amplifier_target_from_row_is_none_when_absent():
    evaluator = evolution_common.TransientGainEvaluator()
    assert evaluator.target_from_row({'desired_gain': ''}) is None
    assert evaluator.target_from_row({}) is None


class _FailingAmp:
    """Stub circuit that always fails, without touching real SPICE."""

    def configure_capacitors(self, *args, **kwargs):
        pass

    def configure_resistors(self, *args, **kwargs):
        raise RuntimeError('boom')

    def transient_analysis(self, *args, **kwargs):
        raise AssertionError('should not be reached — configure_resistors fails first')


def _failing_spec():
    return evolution_common.CircuitSpec(
        circuit_name='fake',
        display_name='Fake Amp',
        circuit_factory=_FailingAmp,
        num_genes=4,
        resistor_mapper=lambda solution: list(solution),
        default_population=4,
        default_generations=10,
    )


def test_fitness_func_failure_returns_minimal_fitness_and_is_tracked():
    run = evolution_common.EvolutionRun(_failing_spec(), target=10, exec_counter=1, seed=1)

    fitness = run.fitness_func(ga_instance=_FakeGA(), solution=[1, 2, 3, 4], solution_idx=0)

    assert fitness == evolution_common._MIN_FITNESS
    assert run._eval_failures == 1
    assert isinstance(run._last_failure_exc, RuntimeError)


def test_fitness_func_failure_count_accumulates():
    run = evolution_common.EvolutionRun(_failing_spec(), target=10, exec_counter=1, seed=1)

    for _ in range(3):
        run.fitness_func(ga_instance=_FakeGA(), solution=[1, 2, 3, 4], solution_idx=0)

    assert run._eval_failures == 3


def test_4r_resistor_mapper_uses_single_100k_pot():
    import amp_4r_evolution as amp4

    pot_100k = components.DigitalPot(100000, 100)
    solution = [1, 50, 100, 25]

    result = amp4.resistor_mapper(solution)

    assert result == [pot_100k.get_resistance(v) for v in solution]


def test_8r_resistor_mapper_alternates_100k_and_10k_pots():
    import amp_8r_evolution as amp8

    pot_100k = components.DigitalPot(100000, 100)
    pot_10k = components.DigitalPot(10000, 100)
    solution = [1, 2, 3, 4, 5, 6, 7, 8]

    result = amp8.resistor_mapper(solution)

    expected = [
        pot_100k.get_resistance(1), pot_10k.get_resistance(2),
        pot_100k.get_resistance(3), pot_10k.get_resistance(4),
        pot_100k.get_resistance(5), pot_10k.get_resistance(6),
        pot_100k.get_resistance(7), pot_10k.get_resistance(8),
    ]
    assert result == expected
