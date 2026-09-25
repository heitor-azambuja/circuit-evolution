"""
The paper's experiment campaign: GA versus the analytical design, both circuits.

Runs every (variant, target, seed) cell of one circuit family and records the
full result of each run, not just its fitness. The median is the comparison that
counts: the GA's *best* exceeds the analytical design in nearly every cell, so
reporting best-of-N would let it "win" everywhere and say nothing.

Each run gets its own worker process, which is then discarded. PySpice leaks
about 47 kB per simulation and never returns it, so a run costs a few hundred
megabytes that only a process exit reclaims.

Early stopping is off by default and should stay off here. At a patience of 50 it
halves the generations but leaves 28% of runs below 99% of the fitness they would
have reached — a bias that lands unevenly across cells.

Split across machines with --shard, which partitions the job list by index:

    python campaign.py --family filter --seeds 100 --shard 1/3 --data-csv m1.csv
    python campaign.py --family filter --seeds 100 --shard 2/3 --data-csv m2.csv

Each shard writes a disjoint set of rows; concatenate the CSVs when they finish.
"""
import argparse
import json
import multiprocessing
import resource

import numpy as np

_ADDRESS_SPACE_MB = 3000

FAMILIES = {
    'filter': {'modules': ('filter_sk4_evolution', 'filter_sk8_evolution')},
    'amp': {'modules': ('amp_4r_evolution', 'amp_8r_evolution')},
}
VARIANTS = ('4R', '8R')


def _limit_memory() -> None:
    limit = _ADDRESS_SPACE_MB * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def _module(family: str, variant: str):
    import importlib
    return importlib.import_module(FAMILIES[family]['modules'][VARIANTS.index(variant)])


def targets_of(family: str) -> list:
    if family == 'filter':
        import filter_targets
        return list(filter_targets.TARGETS)
    return [5, 10, 15, 20]


def baseline_taps(family: str, variant: str, target) -> list:
    """Taps the analytical design picks, knowing only nominal component values."""
    module = _module(family, variant)
    if family == 'filter':
        import filter_baseline
        return filter_baseline.design(module, target)['taps']

    import amp_design
    # quantize_to_series_pots is generic potentiometer arithmetic that happens to
    # live in filter_design; the 8R amplifier needs exactly the same thing.
    import components
    import filter_design

    result = amp_design.design(target)
    pot = components.DigitalPot(*module.POT_SPECS[0])
    r1, r2 = amp_design.divider(result['base_voltage'], pot)
    resistances = [r1, r2, result['resistances'][2], result['resistances'][3]]

    if len(module.POT_SPECS) == 4:
        return amp_design.quantize(resistances, pot)
    fine = components.DigitalPot(*module.POT_SPECS[1])
    return filter_design.quantize_to_series_pots(resistances, pot, fine)


def _run_one(job) -> dict:
    family, variant, target, seed, out_dir, patience = job

    import evolution_common
    spec = _module(family, variant).SPEC
    run = evolution_common.EvolutionRun(spec, target=target, exec_counter=seed, seed=seed)
    data = run.evolve(spec.default_generations, spec.default_population,
                      auto_plots=False, out_dir=out_dir, patience=patience)

    # One extra evaluation buys every result metric for the winning solution.
    # Without it the campaign records a fitness and a set of taps, and recovering
    # the rest means re-simulating all of it afterwards.
    solution = json.loads(data['solution'])
    metrics = spec.evaluator.metrics(run.circuit, spec.resistor_mapper(solution), target)
    data.update(spec.evaluator.metric_fields(metrics))
    data['family'] = family
    data['variant'] = variant
    data['target'] = str(target)
    return data


def run(family: str, seeds: int, out_dir: str, workers: int, patience,
        data_csv: str, shard: tuple) -> list:
    import data_parse

    jobs = [(family, variant, target, seed, out_dir, patience)
            for variant in VARIANTS for target in targets_of(family)
            for seed in range(1, seeds + 1)]
    index, total = shard
    jobs = jobs[index - 1::total]

    results = []
    context = multiprocessing.get_context('spawn')
    with context.Pool(processes=workers, maxtasksperchild=1,
                      initializer=_limit_memory) as pool:
        for done, data in enumerate(pool.imap_unordered(_run_one, jobs), start=1):
            results.append(data)
            if data_csv:
                data_parse.dump_json_to_csv(data_csv, data)
            # PySpice configures the root logger on import and swallows INFO from
            # anywhere else, so progress goes straight to stdout.
            print(f"  [{done}/{len(jobs)}] {data['variant']} {data['target']} "
                  f"seed={data['seed']} fitness={data['solution_fitness']:.6f} "
                  f"gen={data['generations_completed']}", flush=True)
    return results


def summarize(family: str, results: list) -> None:
    cells = sorted({(r['variant'], r['target']) for r in results})
    print()
    for variant, target in cells:
        rows = [r for r in results if r['variant'] == variant and r['target'] == target]
        fitnesses = np.array([r['solution_fitness'] for r in rows])
        native = type(targets_of(family)[0])(target)
        spec = _module(family, variant).SPEC
        circuit = spec.circuit_factory()
        if spec.setup_hook is not None:
            spec.setup_hook(circuit, native)
        taps = baseline_taps(family, variant, native)
        baseline = spec.evaluator.fitness(circuit, spec.resistor_mapper(taps), native)
        print(f'{variant} {str(target):18s} n={len(rows):3d} baseline={baseline:.4f} | '
              f'best={fitnesses.max():.4f} median={np.median(fitnesses):.4f} '
              f'worst={fitnesses.min():.4f} | GA vence '
              f'{int((fitnesses > baseline).sum())}/{len(rows)}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-f', '--family', choices=sorted(FAMILIES), required=True)
    parser.add_argument('-s', '--seeds', type=int, default=100)
    parser.add_argument('-w', '--workers', type=int, default=6)
    parser.add_argument('--patience', type=int, default=None,
                        help='Off by default, and it should stay off here')
    parser.add_argument('--out-dir', default='simulations')
    parser.add_argument('--data-csv')
    parser.add_argument('--out')
    parser.add_argument('--shard', default='1/1',
                        help='i/N — run only the i-th of N disjoint slices')
    args = parser.parse_args()

    index, total = (int(part) for part in args.shard.split('/'))
    if not 1 <= index <= total:
        parser.error(f'shard {args.shard} is out of range')

    results = run(args.family, args.seeds, args.out_dir, args.workers,
                  args.patience, args.data_csv, (index, total))
    summarize(args.family, results)

    if args.out:
        with open(args.out, 'w') as handle:
            json.dump(results, handle, indent=2, default=str)


if __name__ == '__main__':
    main()
