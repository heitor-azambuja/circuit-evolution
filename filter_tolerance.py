"""
Component tolerance: what happens when the board is not the one on the datasheet.

Every result so far assumes exact nominal components, which is an idealisation
that flatters the analytical design. Real parts deviate, and the two methods
answer that deviation very differently:

* The analytical design computes its resistors from *nominal* values, because
  that is all a designer has. If this board's pot is 11.5k rather than 10k, every
  tap it picked is off by fifteen percent and nothing in the procedure can know.
* The GA optimises against the response of *this* board. It never needs to know
  what the parts are nominally worth.

That is the production argument for on-device search: you cannot hand-calibrate
every unit, and a GA calibrates itself.

Tolerances are multiplicative deviations drawn once per board and then fixed for
that board's life. Redrawing them per evaluation would model noise, not spread:
the objective would stop being stationary and there would be nothing to converge
to.

Only the potentiometer's end-to-end resistance deviates, not its tap ratio — the
taps are matched segments on one die, so tap 50 gives half of whatever the part
actually measures. Modelling it this way is both more faithful and more
conservative, since it leaves the analytical design fewer error sources to trip
over.

The distribution is an assumption to declare: uniform within the limits. Vendors
guarantee a bound, not a shape, and real spreads are often bimodal because parts
get sorted by bin.

    python filter_tolerance.py --boards 20 --target chebyshev_1000
"""
import argparse
import dataclasses
import json
import multiprocessing
import resource

import numpy as np

# E12 values are spaced about 20% apart precisely so that +/-10% bands tile the
# range without a gap, so the capacitor tolerance follows from choosing E12.
CAPACITOR_TOLERANCE = 0.10
# X9C datasheet (Renesas FN8222): "End-to-End Resistance Variation, -20 to +20 %".
POT_TOLERANCE = 0.20
# Same datasheet: wiper resistance 40 ohm typical, 100 ohm max. In rheostat mode
# it adds in series, so it is a fixed offset the analytical design never accounts
# for -- on the 1k part that offset is four tap steps.
WIPER_RESISTANCE_TYP = 40.0
WIPER_RESISTANCE_MAX = 100.0

_WORKERS = 3
_ADDRESS_SPACE_MB = 2000


def _limit_memory() -> None:
    limit = _ADDRESS_SPACE_MB * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


@dataclasses.dataclass(frozen=True)
class Board:
    """One physical unit: deviations drawn once and then fixed for its life."""
    index: int
    pot_scale: tuple
    pot_wiper: tuple
    capacitor_scale: tuple


def draw_board(index: int, pot_count: int, capacitor_count: int = 4,
               pot_tolerance: float = POT_TOLERANCE,
               capacitor_tolerance: float = CAPACITOR_TOLERANCE) -> Board:
    """Deviations for one board. Seeded by index so boards are reproducible."""
    rng = np.random.default_rng(index)
    return Board(
        index=index,
        pot_scale=tuple(rng.uniform(1 - pot_tolerance, 1 + pot_tolerance, pot_count)),
        # The datasheet bounds the wiper but gives no distribution, so it is drawn
        # across its range like everything else here.
        pot_wiper=tuple(rng.uniform(0.0, WIPER_RESISTANCE_MAX, pot_count)),
        capacitor_scale=tuple(rng.uniform(1 - capacitor_tolerance,
                                          1 + capacitor_tolerance, capacitor_count)),
    )


def board_spec(module, board: Board):
    """The module's CircuitSpec as it behaves on this particular board."""
    import components
    from filter_targets import TARGETS

    pots = [components.DigitalPot(full_scale * scale, taps, wiper_resistance=wiper)
            for (full_scale, taps), scale, wiper
            in zip(module.POT_SPECS, board.pot_scale, board.pot_wiper)]

    def configure(circuit, target) -> None:
        nominal = TARGETS[target]['capacitors_nf']
        circuit.configure_capacitors(*[c * s for c, s in zip(nominal, board.capacitor_scale)])

    return dataclasses.replace(module.SPEC,
                               resistor_mapper=module.make_resistor_mapper(pots),
                               setup_hook=configure)


def _evaluate(spec, target, taps) -> float:
    circuit = spec.circuit_factory()
    spec.setup_hook(circuit, target)
    return float(spec.evaluator.fitness(circuit, spec.resistor_mapper(taps), target))


def _run_board(job) -> dict:
    variant, target, board_index, seeds = job

    import evolution_common
    import filter_baseline
    import filter_sk4_evolution
    import filter_sk8_evolution

    module = {'4R': filter_sk4_evolution, '8R': filter_sk8_evolution}[variant]
    board = draw_board(board_index, len(module.POT_SPECS))
    spec = board_spec(module, board)

    # The designer only ever had nominal values, so the taps come from those.
    nominal_taps = filter_baseline.design(module, target)['taps']
    baseline = _evaluate(spec, target, nominal_taps)

    fitnesses = []
    for seed in range(1, seeds + 1):
        run = evolution_common.EvolutionRun(spec, target=target,
                                            exec_counter=board_index * 100 + seed,
                                            seed=board_index * 1000 + seed)
        data = run.evolve(spec.default_generations, spec.default_population,
                          auto_plots=False, out_dir='simulations/tolerance')
        fitnesses.append(data['solution_fitness'])

    return {
        'variant': variant,
        'target': target,
        'board': board_index,
        'pot_scale': list(board.pot_scale),
        'pot_wiper': list(board.pot_wiper),
        'capacitor_scale': list(board.capacitor_scale),
        'baseline_fitness': baseline,
        'ga_fitness': fitnesses,
        'ga_median': float(np.median(fitnesses)),
    }


def run(variants, targets, boards: int, seeds: int) -> list:
    jobs = [(variant, target, index, seeds)
            for variant in variants for target in targets
            for index in range(1, boards + 1)]

    results = []
    context = multiprocessing.get_context('spawn')
    with context.Pool(processes=_WORKERS, maxtasksperchild=1,
                      initializer=_limit_memory) as pool:
        for done, row in enumerate(pool.imap_unordered(_run_board, jobs), start=1):
            results.append(row)
            print(f"  [{done}/{len(jobs)}] {row['variant']} {row['target']} "
                  f"placa {row['board']:2d} | baseline={row['baseline_fitness']:.4f} "
                  f"AG={row['ga_median']:.4f}", flush=True)
    return results


def summarize(results: list) -> None:
    rmse = lambda f: 1.0 / f - 1.0
    keys = sorted({(r['variant'], r['target']) for r in results})
    print()
    for variant, target in keys:
        rows = [r for r in results if r['variant'] == variant and r['target'] == target]
        base = np.array([rmse(r['baseline_fitness']) for r in rows])
        eva = np.array([rmse(r['ga_median']) for r in rows])
        wins = sum(1 for r in rows if r['ga_median'] > r['baseline_fitness'])
        print(f'{variant} {target:18s} placas={len(rows)}')
        print(f'    baseline  rmse mediana {np.median(base):.4f} dB  '
              f'pior {base.max():.4f} dB')
        print(f'    AG        rmse mediana {np.median(eva):.4f} dB  '
              f'pior {eva.max():.4f} dB')
        print(f'    AG melhor que o baseline em {wins}/{len(rows)} placas')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-b', '--boards', type=int, default=20)
    parser.add_argument('-s', '--seeds', type=int, default=4)
    parser.add_argument('--variant', action='append', choices=('4R', '8R'))
    parser.add_argument('--target', action='append')
    parser.add_argument('--out')
    args = parser.parse_args()

    import filter_targets
    variants = args.variant or ['4R', '8R']
    targets = args.target or ['chebyshev_1000']
    unknown = set(targets) - set(filter_targets.TARGETS)
    if unknown:
        parser.error(f'unknown target(s): {sorted(unknown)}')

    results = run(variants, targets, args.boards, args.seeds)
    summarize(results)

    if args.out:
        with open(args.out, 'w') as handle:
            json.dump(results, handle, indent=2)


if __name__ == '__main__':
    main()
