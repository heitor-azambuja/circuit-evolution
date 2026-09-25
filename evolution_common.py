"""
Shared GA/plotting logic for the amplifier evolution scripts.

Each `amp_*r_evolution.py` script defines a `CircuitSpec` describing its
circuit and resistor layout, then calls `run_cli(spec)`. All the fitness
evaluation, early stopping, plotting and CSV/JSON output logic lives here so
it isn't duplicated (and doesn't drift) between the 4R and 8R scripts.
"""
import argparse
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pygad
import scipy.spatial

import data_parse

logger = logging.getLogger()

# Simulation time step (s) used for both fitness evaluation and plots.
_STEP_TIME_FINE = 0.000005

_MIN_FITNESS = 1e-10
_MAX_FITNESS = 1e10


# Most mutations nudge a tap by at most this many positions; the rest jump anywhere.
_LOCAL_MUTATION_STEP = 3
_LOCAL_MUTATION_SHARE = 0.8


def tap_mutation(offspring, ga_instance):
    """Mutate potentiometer taps with small local steps, occasionally jumping far.

    pygad's built-in random mutation replaces a gene with a value drawn uniformly
    from its gene_space, so it can never refine a solution by one tap. These
    circuits need exactly that: near an optimum a single tap of error costs more
    fitness than the whole margin between a good and a great solution, so a GA
    that can only jump blindly plateaus short of the analytical design. Keeping
    the uniform jump as a minority case preserves exploration.
    """
    bounds = [(min(space), max(space)) for space in ga_instance.gene_space]
    probability = ga_instance.mutation_probability

    for individual in offspring:
        for gene, (low, high) in enumerate(bounds):
            if np.random.random() >= probability:
                continue
            if np.random.random() < _LOCAL_MUTATION_SHARE:
                step = np.random.randint(1, _LOCAL_MUTATION_STEP + 1)
                value = individual[gene] + step * np.random.choice([-1, 1])
            else:
                value = np.random.randint(low, high + 1)
            individual[gene] = min(max(value, low), high)

    return offspring


class TransientGainEvaluator:
    """Amplifier objective: match an inverted, amplified copy of the input, in the time domain.

    An evaluator owns everything about *how quality is measured* — which analysis to
    run, the objective itself, the plot, and how the target is named in filenames and
    CSV columns. `EvolutionRun` owns the GA plumbing and knows none of it, which is
    what lets a frequency-domain target coexist with this one.

    The interface:
        fitness(circuit, resistances, target) -> float
        metrics(circuit, resistances, target) -> dict
        plot(metrics, save_path, title) -> None
        target_fields(target) -> dict    # target columns seeded into the CSV row
        metric_fields(metrics) -> dict   # result columns added to the CSV row
        target_slug(target) -> str       # filename fragment
        target_label(target) -> str      # plot title fragment
    """

    def fitness(self, circuit, resistances, target) -> float:
        circuit.configure_resistors(resistances)
        analysis = circuit.transient_analysis(step_time=_STEP_TIME_FINE)
        output = np.array(analysis.out)
        desired = -target * np.array(analysis['in'])
        steady_start = int(0.25 * len(output))
        distance = scipy.spatial.distance.euclidean(output[steady_start:],
                                                    desired[steady_start:])
        return 1.0 / distance if distance != 0 else _MAX_FITNESS

    def metrics(self, circuit, resistances, target) -> dict:
        return compute_metrics(circuit, resistances, target)

    def plot(self, metrics: dict, save_path: str, title: str) -> None:
        plot_waveforms(metrics, save_path, title)

    def target_fields(self, target) -> dict:
        return {'desired_gain': target}

    def metric_fields(self, metrics: dict) -> dict:
        fields = {key: metrics[key] for key in (
            'max_voltage', 'min_voltage', 'max_error', 'max_error_percent',
            'avg_error', 'avg_error_percent')}
        fields['resistors'] = metrics['resistances']
        return fields

    def target_slug(self, target) -> str:
        return f'gain{target}'

    def target_label(self, target) -> str:
        return f'Gain = {target}'

    def target_from_row(self, row: dict):
        """Recover the target from a saved CSV row, for regenerating plots."""
        raw = row.get('desired_gain')
        if raw in (None, ''):
            return None
        value = float(raw)
        # Keep whole gains integral so regenerated filenames match the originals.
        return int(value) if value.is_integer() else value


@dataclass
class CircuitSpec:
    circuit_name: str
    display_name: str
    circuit_factory: Callable[[], object]
    num_genes: int
    resistor_mapper: Callable[[List[int]], List[float]]
    default_population: int
    default_generations: int = 400
    evaluator: object = field(default_factory=TransientGainEvaluator)
    # Called once per run as setup_hook(circuit, target) — e.g. to fix capacitors,
    # which for a filter depend on which response is being targeted.
    setup_hook: Optional[Callable[[object, object], None]] = None
    gene_space: Optional[list] = None
    # Set to 'random' for pygad's built-in uniform-replacement mutation.
    mutation_type: object = tap_mutation


def compute_metrics(amp, resistances: list, desired_gain: float,
                     step_time: float = _STEP_TIME_FINE) -> dict:
    """Run a transient simulation for `resistances` and return waveform arrays + error metrics."""
    amp.configure_resistors(resistances)
    analysis = amp.transient_analysis(step_time=step_time)
    output = 100 * np.array(analysis.out)
    desired = 100 * (-desired_gain) * np.array(analysis['in'])
    time_us = np.array(analysis.time) * 1_000_000

    max_error_mv = float(np.max(np.abs(output - desired)))
    avg_error_mv = float(np.mean(np.abs(output - desired)))

    return {
        'resistances': resistances,
        'output': output,
        'desired': desired,
        'time_us': time_us,
        'max_voltage': float(np.max(output)),
        'min_voltage': float(np.min(output)),
        'max_error': max_error_mv,
        'max_error_percent': max_error_mv / desired_gain * 100,
        'avg_error': avg_error_mv,
        'avg_error_percent': avg_error_mv / desired_gain * 100,
    }


def plot_waveforms(metrics: dict, save_path: str, title: str) -> None:
    """Save a waveform comparison plot (output vs desired + error panel) to save_path."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 7),
                              gridspec_kw={'height_ratios': [3, 1]},
                              constrained_layout=True)

    ax = axes[0]
    t = metrics['time_us']
    out = metrics['output']
    des = metrics['desired']

    ax.plot(t, des, color='#2ca02c', linewidth=1.8, linestyle='--', label='desired', zorder=3)
    ax.plot(t, out, color='#1f77b4', linewidth=1.8, label='output', zorder=4)
    ax.fill_between(t, out, des, alpha=0.18, color='#d62728', label='error')

    ax.set_xlabel('time [µs]', fontsize=11)
    ax.set_ylabel('voltage [mV]', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)

    res_str = ', '.join(f'{r/1000:.1f}k' for r in metrics['resistances'])
    annotation = (
        f"avg err: {metrics['avg_error_percent']:.2f}%\n"
        f"max err: {metrics['max_error_percent']:.2f}%\n"
        f"R: [{res_str}] Ω"
    )
    ax.text(0.01, 0.97, annotation, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='#aaaaaa'))

    axe = axes[1]
    axe.fill_between(t, np.abs(out - des), color='#d62728', alpha=0.5, linewidth=0)
    axe.plot(t, np.abs(out - des), color='#d62728', linewidth=1.0)
    axe.set_xlabel('time [µs]', fontsize=11)
    axe.set_ylabel('|error| [mV]', fontsize=11)
    axe.grid(True, linestyle=':', alpha=0.6)

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


class EvolutionRun:
    """Holds the mutable state for a single GA run (one gain, one repetition).

    Replaces what used to be module-level globals (amp, data_json, ga_seed,
    exec_counter, desired_gain, generations_completed, early-stop counters)
    so runs no longer share state and can be unit tested in isolation.
    """

    def __init__(self, spec: CircuitSpec, target, exec_counter: int,
                 seed: Optional[int] = None):
        self.spec = spec
        self.target = target
        self.exec_counter = exec_counter
        self.seed = seed
        self.circuit = spec.circuit_factory()
        if spec.setup_hook is not None:
            spec.setup_hook(self.circuit, target)

        self.generations_completed = 0
        self.patience = None  # set in evolve()
        self._eval_failures = 0
        self._last_failure_exc = None
        self._no_improve_count = 0
        self._best_fitness_seen = -np.inf

        self.data_json = {
            'ckt_name': spec.circuit_name,
            **spec.evaluator.target_fields(target),
            'exec_counter': exec_counter,
            'seed': seed,
            'run_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
        }

    def fitness_func(self, ga_instance, solution, solution_idx) -> float:
        try:
            resistances = self.spec.resistor_mapper(solution)
            fitness = self.spec.evaluator.fitness(self.circuit, resistances, self.target)

            current_generation = ga_instance.generations_completed
            if current_generation > self.generations_completed:
                logger.debug(f'Completed generation {current_generation}')
                self.generations_completed = current_generation
            return fitness
        except Exception as exc:
            # Expected for some resistor combinations (e.g. SPICE convergence
            # failures); tracked and surfaced in evolve() so it isn't silent.
            self._eval_failures += 1
            self._last_failure_exc = exc
            logger.debug(f'Fitness evaluation failed: {exc}')
            return _MIN_FITNESS

    def compute_metrics(self, solution) -> dict:
        resistances = self.spec.resistor_mapper(solution)
        return self.spec.evaluator.metrics(self.circuit, resistances, self.target)

    def plot_solution(self, solution, out_dir: str = 'simulations') -> dict:
        """Compute metrics, store them in data_json, and save the result plot."""
        evaluator = self.spec.evaluator
        metrics = self.compute_metrics(solution)
        self.data_json.update(evaluator.metric_fields(metrics))

        title = (
            f'{self.spec.display_name} — {evaluator.target_label(self.target)}   '
            f'(exec {self.exec_counter}, seed {self.seed})'
        )
        path = (f'{out_dir}/{self.spec.circuit_name}'
                f'_{evaluator.target_slug(self.target)}_execution{self.exec_counter}.png')
        evaluator.plot(metrics, path, title)
        return metrics

    def on_generation(self, ga_instance):
        current_best = ga_instance.best_solution()[1]
        if current_best > self._best_fitness_seen:
            self._best_fitness_seen = current_best
            self._no_improve_count = 0
        else:
            self._no_improve_count += 1
        if self._no_improve_count >= self.patience:
            return 'stop'

    def evolve(self, generations: int, population: int, mutation_probability: float = 0.1,
               auto_plots: bool = False, out_dir: str = 'simulations') -> dict:
        self.patience = generations
        self._no_improve_count = 0
        self._best_fitness_seen = -np.inf

        start_time = time.perf_counter()
        num_parents_mating = max(2, population // 2)
        gene_space = self.spec.gene_space or [range(1, 101)] * self.spec.num_genes
        ga_instance = pygad.GA(num_generations=generations,
                                num_parents_mating=num_parents_mating,
                                fitness_func=self.fitness_func,
                                sol_per_pop=population,
                                num_genes=self.spec.num_genes,
                                gene_type=int,
                                gene_space=gene_space,
                                init_range_low=1,
                                init_range_high=100,
                                parent_selection_type="tournament",
                                keep_parents=1,
                                crossover_type="single_point",
                                mutation_type=self.spec.mutation_type,
                                mutation_probability=mutation_probability,
                                on_generation=self.on_generation,
                                random_seed=self.seed)

        ga_instance.run()

        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        runtime_s = time.perf_counter() - start_time

        self.data_json['generations'] = generations
        self.data_json['population'] = population
        self.data_json['solution'] = json.dumps(solution.tolist())
        self.data_json['solution_fitness'] = solution_fitness
        self.data_json['best_generation'] = getattr(ga_instance, 'best_solution_generation', None)
        self.data_json['runtime_s'] = runtime_s

        if self._eval_failures:
            self.data_json['fitness_eval_failures'] = self._eval_failures
            logger.warning(
                f'{self._eval_failures} fitness evaluation(s) raised an exception during this run '
                f'(returned minimal fitness instead of crashing). Last error: {self._last_failure_exc}'
            )

        # Save per-generation fitness history so plots can be regenerated later.
        target_slug = self.spec.evaluator.target_slug(self.target)
        fitness_history_path = (
            f'{out_dir}/fitness_history_{self.spec.circuit_name}'
            f'_{target_slug}_execution{self.exec_counter}.json'
        )
        hist = getattr(ga_instance, 'best_solutions_fitness', None)
        if hist is None:
            hist_list = []
        elif hasattr(hist, 'tolist'):
            hist_list = hist.tolist()
        else:
            hist_list = list(hist)

        with open(fitness_history_path, 'w') as fh:
            json.dump({
                'run_id': self.data_json.get('run_id', ''),
                'ckt_name': self.spec.circuit_name,
                **self.spec.evaluator.target_fields(self.target),
                'target_slug': target_slug,
                'exec_counter': self.exec_counter,
                'seed': self.seed,
                'best_solutions_fitness': hist_list,
            }, fh)

        logger.info(f'Parameters of the best solution : {solution}')
        logger.info(f'Fitness value of the best solution = {solution_fitness}')

        # Optionally generate plots (disabled by default to speed batch runs).
        if auto_plots:
            fitness_path = (
                f'{out_dir}/fitness_{self.spec.circuit_name}'
                f'_{target_slug}_execution{self.exec_counter}.png'
            )
            fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
            ax.plot(ga_instance.best_solutions_fitness, linewidth=2, color='#1f77b4')
            ax.set_xlabel('Generation', fontsize=11)
            ax.set_ylabel('Fitness', fontsize=11)
            ax.set_title(f'Fitness — {self.spec.circuit_name} {target_slug} '
                         f'exec={self.exec_counter}', fontsize=11)
            ax.grid(True, linestyle=':', alpha=0.6)
            fig.savefig(fitness_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            self.plot_solution(solution, out_dir=out_dir)

        return self.data_json


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--plots', dest='plots', action='store_true', default=False,
                         help='Enable saving PNG plots after run')
    parser.add_argument('-g', '--generations', type=int, help='Number of generations')
    parser.add_argument('-p', '--population', type=int, help='Population size')
    parser.add_argument('-r', '--repetitions', type=int, help='Repetitions per gain')
    parser.add_argument('--exec-counter', type=int, help='Execution counter override')
    parser.add_argument('--seed', type=int, help='RNG seed')
    parser.add_argument('--gain', type=int, help='Run only for this gain')
    parser.add_argument('--target', help='Run only for this target (non-numeric targets)')
    return parser


def run_cli(spec: CircuitSpec, target_list: Optional[List] = None,
            repetitions: int = 20, data_csv: str = 'simulations/data.csv') -> None:
    if target_list is None:
        target_list = [5, 10, 15, 20]

    parser = build_arg_parser(f'Run {spec.circuit_name} GA experiments')
    args = parser.parse_args()

    auto_plots = bool(args.plots)
    generations = args.generations or spec.default_generations
    population = args.population or spec.default_population
    if args.repetitions:
        repetitions = args.repetitions
    if args.seed:
        np.random.seed(args.seed)
    if args.gain is not None:
        target_list = [args.gain]
    if args.target is not None:
        target_list = [args.target]

    for target in target_list:
        for i in range(repetitions):
            exec_counter = args.exec_counter if args.exec_counter is not None else i + 1
            logger.info(f'Running evolution for target = {target} ({i + 1}/{repetitions})')

            seed = int(np.random.randint(0, 2**31 - 1))
            np.random.seed(seed)

            run = EvolutionRun(spec, target=target, exec_counter=exec_counter, seed=seed)
            data_json = run.evolve(generations, population, auto_plots=auto_plots)
            data_parse.dump_json_to_csv(data_csv, data_json)
