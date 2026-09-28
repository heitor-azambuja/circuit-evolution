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
The CSV is the only output, and it is appended run by run: an interrupted
campaign keeps everything it had already finished.

--variant and --population exist for one reason: the two variants do not default
to the same search budget. 4R evolves 4 genes with a population of 20 and 8R
evolves 8 with a population of 40, so a straight 4R-vs-8R result confounds
resistor resolution with the number of evaluations each variant was given. To
separate them, re-run one variant at the other's population and compare against
its own earlier rows:

    python campaign.py --family filter --variant 4R --population 40 \
        --seeds 100 --data-csv control_4r_pop40.csv --out-dir simulations/control

Every row records the population it ran with, so the control and the original are
distinguishable in the CSV without tracking which file came from which command.
"""
import argparse
import glob
import json
import multiprocessing
import os
import resource

import numpy as np

# Address space, not resident memory: numpy and scipy reserve large virtual
# mappings, so this has to be generous or a worker can fail during import. A
# tolerance study deadlocked at 2 GB. It exists only to stop a runaway leak from
# taking the host down -- a run's actual footprint is well under a gigabyte.
_ADDRESS_SPACE_MB = 12000

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
    family, variant, target, seed, out_dir, patience, population, generations = job

    import evolution_common
    spec = _module(family, variant).SPEC
    run = evolution_common.EvolutionRun(spec, target=target, exec_counter=seed, seed=seed)
    data = run.evolve(generations or spec.default_generations,
                      population or spec.default_population,
                      auto_plots=False, out_dir=out_dir, patience=patience)

    # One extra evaluation buys every result metric for the winning solution.
    # Without it the campaign records a fitness and a set of taps, and recovering
    # the rest means re-simulating all of it afterwards.
    solution = json.loads(data['solution'])
    metrics = spec.evaluator.metrics(run.circuit, spec.resistor_mapper(solution), target)
    data.update(spec.evaluator.metric_fields(metrics))
    # Always present, so every row carries the same columns; the CSV appends
    # against the header already on disk and would drop a key that shows up late.
    data.setdefault('fitness_eval_failures', 0)
    data['family'] = family
    data['variant'] = variant
    data['target'] = str(target)
    return data


def job_list(family: str, seeds: int, out_dir: str, patience, variants: tuple = VARIANTS,
             population: int = None, generations: int = None,
             targets: tuple = None) -> list:
    """Every run this campaign will perform, in the order --shard partitions.

    Separate from `run` so it can be inspected without starting a pool -- the job
    tuple is positional and `_run_one` unpacks it, so the two have to agree.
    """
    return [(family, variant, target, seed, out_dir, patience, population, generations)
            for variant in variants
            for target in (targets or targets_of(family))
            for seed in range(1, seeds + 1)]


def run(family: str, seeds: int, out_dir: str, workers: int, patience,
        data_csv: str, shard: tuple, variants: tuple = VARIANTS,
        population: int = None, generations: int = None,
        targets: tuple = None) -> list:
    import data_parse

    jobs = job_list(family, seeds, out_dir, patience, variants=variants,
                    population=population, generations=generations, targets=targets)
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


def warn_about_history_collisions(family: str, variants: tuple, out_dir: str,
                                  population: int, targets: tuple = None) -> list:
    """Warn when a re-run would overwrite fitness histories already in `out_dir`.

    The per-run history filename is built from circuit, target and seed only -- not
    from the population -- so a control run at a different population writes over
    the histories of the campaign it is meant to be compared against. The CSV rows
    are safe either way, since the campaign appends those, but the histories are the
    record behind every convergence plot. Returns the paths that would be lost.

    It fires whenever --population differs from the variant's own default and any
    matching history exists, which is deliberately conservative: nothing on disk
    records the population a history came from, so the alternative to warning too
    often is overwriting silently.
    """
    if population is None:
        return []

    doomed = []
    for variant in variants:
        spec = _module(family, variant).SPEC
        if population == spec.default_population:
            continue
        for target in (targets or targets_of(family)):
            slug = spec.evaluator.target_slug(target)
            pattern = os.path.join(
                out_dir, f'fitness_history_{spec.circuit_name}_{slug}_execution*.json')
            doomed.extend(glob.glob(pattern))

    if doomed:
        # Nothing on disk records the population a history was produced with, so this
        # cannot tell whether the existing files came from a different one. It reports
        # what it does know: these runs will overwrite those files.
        print(f'WARNING: this run will overwrite {len(doomed)} fitness history '
              f'file(s) already in {out_dir}/. The filenames are built from circuit, '
              f'target and seed only, so a run at population {population} cannot '
              f'coexist there with whatever wrote them.', flush=True)
        print(f'         Pass a separate --out-dir (for example '
              f'{out_dir.rstrip("/")}-pop{population}) to keep both.', flush=True)
    return doomed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-f', '--family', choices=sorted(FAMILIES), required=True)
    parser.add_argument('-s', '--seeds', type=int, default=100)
    parser.add_argument('-w', '--workers', type=int, default=6)
    parser.add_argument('--variant', choices=VARIANTS, action='append',
                        help='Run only this variant; repeat for several. Default: both')
    parser.add_argument('--population', type=int,
                        help='Override the population both variants would otherwise '
                             'take from their own spec (4R: 20, 8R: 40). Use it to '
                             'give the variants a matched search budget')
    parser.add_argument('--generations', type=int,
                        help='Override the generations taken from the spec (400). The '
                             'other half of the search budget')
    parser.add_argument('--target', action='append',
                        help='Run only this target; repeat for several. Default: all')
    parser.add_argument('--patience', type=int, default=None,
                        help='Off by default, and it should stay off here')
    parser.add_argument('--out-dir', default='simulations')
    parser.add_argument('--data-csv', required=True,
                        help='Every run is appended here as it finishes')
    parser.add_argument('--shard', default='1/1',
                        help='i/N — run only the i-th of N disjoint slices')
    args = parser.parse_args()

    index, total = (int(part) for part in args.shard.split('/'))
    if not 1 <= index <= total:
        parser.error(f'shard {args.shard} is out of range')
    if args.population is not None and args.population < 4:
        parser.error('a population below 4 leaves nothing to mate')

    if args.generations is not None and args.generations < 1:
        parser.error('a campaign needs at least one generation')

    variants = tuple(dict.fromkeys(args.variant)) if args.variant else VARIANTS
    known = targets_of(args.family)
    targets = None
    if args.target:
        targets = tuple(dict.fromkeys(type(known[0])(t) for t in args.target))
        unknown = [t for t in targets if t not in known]
        if unknown:
            parser.error(f'unknown {args.family} target(s): {unknown}')
    warn_about_history_collisions(args.family, variants, args.out_dir, args.population,
                                 targets=targets)

    results = run(args.family, args.seeds, args.out_dir, args.workers,
                  args.patience, args.data_csv, (index, total),
                  variants=variants, population=args.population,
                  generations=args.generations, targets=targets)
    summarize(args.family, results)


if __name__ == '__main__':
    main()
