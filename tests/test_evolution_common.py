import components
import evolution_common


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
    run = evolution_common.EvolutionRun(_failing_spec(), desired_gain=10, exec_counter=1, seed=1)

    fitness = run.fitness_func(ga_instance=None, solution=[1, 2, 3, 4], solution_idx=0)

    assert fitness == evolution_common._MIN_FITNESS
    assert run._eval_failures == 1
    assert isinstance(run._last_failure_exc, RuntimeError)


def test_fitness_func_failure_count_accumulates():
    run = evolution_common.EvolutionRun(_failing_spec(), desired_gain=10, exec_counter=1, seed=1)

    for _ in range(3):
        run.fitness_func(ga_instance=None, solution=[1, 2, 3, 4], solution_idx=0)

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
