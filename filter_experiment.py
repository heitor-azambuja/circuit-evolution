"""
The filter experiment: GA versus the analytical design, across the 2x2.

Cells are {4R, 8R} x {Butterworth, Chebyshev}, several seeds each. For every cell
it reports the analytically designed, tap-quantized baseline and the distribution
of GA results against it -- the median being the honest comparison, since taking
the best of N runs would let the GA "win" everywhere.

Each GA run executes in its own worker process, which is then discarded. This is
not tidiness: PySpice leaks about 47 kB per simulation and never returns it, so a
run costs a few hundred megabytes that only a process exit reclaims. Sixteen runs
in one process took a development machine down.

    python filter_experiment.py --seeds 8 --out results.json
"""
import argparse
import json
import multiprocessing
import resource

import numpy as np

_WORKERS = 3
_ADDRESS_SPACE_MB = 2000
VARIANTS = ('4R', '8R')


def _limit_memory() -> None:
    """Cap a worker's address space so the leak kills the worker, not the machine."""
    limit = _ADDRESS_SPACE_MB * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def _module(variant: str):
    import filter_sk4_evolution
    import filter_sk8_evolution
    return {'4R': filter_sk4_evolution, '8R': filter_sk8_evolution}[variant]


def _run_one(job) -> tuple:
    variant, target, seed, out_dir = job

    import evolution_common
    spec = _module(variant).SPEC
    run = evolution_common.EvolutionRun(spec, target=target, exec_counter=seed, seed=seed)
    data = run.evolve(spec.default_generations, spec.default_population,
                      auto_plots=False, out_dir=out_dir)
    return variant, target, seed, data['solution_fitness'], data['solution']


def _baseline(variant: str, target: str) -> float:
    import filter_baseline
    return filter_baseline.evaluate(_module(variant), target)['solution_fitness']


def run(seeds: int, out_dir: str) -> list:
    targets = list(_module('4R').TARGETS)
    jobs = [(variant, target, seed, out_dir)
            for variant in VARIANTS for target in targets
            for seed in range(1, seeds + 1)]

    collected = {}
    context = multiprocessing.get_context('spawn')
    with context.Pool(processes=_WORKERS, maxtasksperchild=1,
                      initializer=_limit_memory) as pool:
        for done, (variant, target, seed, fitness, solution) in enumerate(
                pool.imap_unordered(_run_one, jobs), start=1):
            collected.setdefault((variant, target), []).append((fitness, solution))
            # PySpice configures the root logger on import, which swallows
            # INFO from here, so progress goes straight to stdout.
            print(f'  [{done}/{len(jobs)}] {variant} {target} seed={seed} '
                  f'fitness={fitness:.6f}', flush=True)

    results = []
    for variant in VARIANTS:
        for target in targets:
            entries = collected[(variant, target)]
            fitnesses = np.array([f for f, _ in entries])
            baseline = _baseline(variant, target)
            best_index = int(np.argmax(fitnesses))
            results.append({
                'variant': variant,
                'target': target,
                'baseline_fitness': baseline,
                'best_fitness': float(fitnesses.max()),
                'median_fitness': float(np.median(fitnesses)),
                'worst_fitness': float(fitnesses.min()),
                'wins': int((fitnesses > baseline).sum()),
                'runs': len(fitnesses),
                'best_solution': entries[best_index][1],
            })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-s', '--seeds', type=int, default=8)
    parser.add_argument('--out', help='Write results to this JSON file')
    parser.add_argument('--out-dir', default='simulations',
                        help='Where per-run fitness histories are written')
    args = parser.parse_args()

    results = run(args.seeds, args.out_dir)

    print()
    for row in results:
        rmse = lambda f: 1.0 / f - 1.0
        print(f"{row['variant']} {row['target']:12s} "
              f"baseline={row['baseline_fitness']:.4f} | "
              f"best={row['best_fitness']:.4f} median={row['median_fitness']:.4f} "
              f"worst={row['worst_fitness']:.4f} | "
              f"GA vence {row['wins']}/{row['runs']} | "
              f"rmse mediana {rmse(row['median_fitness']):.5f} dB")

    if args.out:
        with open(args.out, 'w') as handle:
            json.dump(results, handle, indent=2)


if __name__ == '__main__':
    main()
