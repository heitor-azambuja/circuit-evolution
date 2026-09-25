"""
Exhaustive global optimum for the 4R filter.

The two Sallen-Key stages are cascaded through the op-amp's output, which the
macromodel drives as an ideal voltage source, so the second stage does not load
the first: the overall response is exactly the product of the two stage
responses. That turns a 100^4 search into 2 x 100^2 simulations plus arithmetic,
and makes the true optimum computable rather than merely estimated.

With the optimum in hand, "the GA beat the analytical design 4 times out of 8"
becomes "the GA reaches 99.x% of what this hardware can do, the analytical design
reaches 97.y%" -- and the analytical design's rank among all 100 million tap
combinations says how much was left on the table.

Only the 4R variant is enumerable. The 8R variant has 100^8 = 10^16 combinations,
which is itself an argument for search: past a certain resolution the optimum
stops being knowable and a GA is the only way to spend the extra taps.

PySpice leaks about 47 kB of memory per simulation and never gives it back, so
the simulation phase runs in worker processes that are recycled every batch and
capped with RLIMIT_AS. Running 20,000 simulations in one process is what took a
machine down during development.

    python filter_optimum.py --target butterworth
"""
import argparse
import itertools
import json
import logging
import multiprocessing
import resource
import time

import numpy as np

logger = logging.getLogger(__name__)

_BATCH = 250          # simulations per worker before it is recycled
_WORKERS = 4          # kept well under the core count; these are memory-bound
_CHUNK = 16           # stage-A rows combined at once; sets peak RAM of the search
_ADDRESS_SPACE_MB = 2000


def _limit_memory() -> None:
    """Cap a worker's address space so a leak kills the worker, not the machine."""
    limit = _ADDRESS_SPACE_MB * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def _simulate_batch(job) -> tuple:
    """Run one batch of tap pairs in a fresh worker. Returns dB magnitudes.

    One AC run yields both stages: V(stagea) is the first stage's transfer
    function (the source is 1 V AC) and V(out)/V(stagea) is the second's.
    """
    target, pairs, start_index = job

    import filter_sk4_evolution as sk4
    pot = sk4.pot_10k
    evaluator = sk4.SPEC.evaluator

    circuit = sk4.SPEC.circuit_factory()
    sk4.SPEC.setup_hook(circuit, target)
    middle = pot.get_resistance(pot.tap_points // 2)
    sweep = dict(start_frequency=evaluator.start_hz, stop_frequency=evaluator.stop_hz,
                 points_per_decade=evaluator.points_per_decade)

    first, second, freqs = [], [], None
    for i, j in pairs:
        circuit.configure_resistors([pot.get_resistance(i), pot.get_resistance(j),
                                     middle, middle])
        analysis = circuit.ac_analysis(**sweep)
        if freqs is None:
            freqs = np.array(analysis.frequency)
        first.append(np.abs(np.array(analysis['stagea'])))

        circuit.configure_resistors([middle, middle,
                                     pot.get_resistance(i), pot.get_resistance(j)])
        analysis = circuit.ac_analysis(**sweep)
        second.append(np.abs(np.array(analysis['out']) / np.array(analysis['stagea'])))

    to_db = lambda values: (20 * np.log10(np.asarray(values, dtype=np.float64))).astype(np.float32)
    return start_index, to_db(first), to_db(second), freqs


def _simulate_all(target: str, pairs: list) -> tuple:
    """Every tap pair for both stages, run across recycled worker processes."""
    jobs = [(target, pairs[i:i + _BATCH], i) for i in range(0, len(pairs), _BATCH)]
    first = np.empty((len(pairs), 0), dtype=np.float32)
    second = None
    freqs = None
    done = 0

    context = multiprocessing.get_context('spawn')
    with context.Pool(processes=_WORKERS, maxtasksperchild=1,
                      initializer=_limit_memory) as pool:
        for start_index, block_first, block_second, block_freqs in pool.imap_unordered(
                _simulate_batch, jobs):
            if second is None:
                freqs = block_freqs
                first = np.empty((len(pairs), len(freqs)), dtype=np.float32)
                second = np.empty_like(first)
            first[start_index:start_index + len(block_first)] = block_first
            second[start_index:start_index + len(block_second)] = block_second
            done += len(block_first)
            logger.info(f'  simulated {done}/{len(pairs)} tap pairs per stage')

    return first, second, freqs


def _check_separability(target: str) -> float:
    """Largest dB by which one stage moves when the other changes. Must be zero."""
    import filter_sk4_evolution as sk4

    pot = sk4.pot_10k
    evaluator = sk4.SPEC.evaluator
    circuit = sk4.SPEC.circuit_factory()
    sk4.SPEC.setup_hook(circuit, target)

    responses = []
    for other in (pot.get_resistance(20), pot.get_resistance(80)):
        circuit.configure_resistors([pot.get_resistance(50), pot.get_resistance(50),
                                     other, other])
        analysis = circuit.ac_analysis(start_frequency=evaluator.start_hz,
                                       stop_frequency=evaluator.stop_hz,
                                       points_per_decade=evaluator.points_per_decade)
        responses.append(20 * np.log10(np.abs(np.array(analysis['stagea']))))

    return float(np.max(np.abs(responses[0] - responses[1])))


def _exact_fitness(target: str, taps: list) -> float:
    """Re-score through the real evaluator, so the headline number is not the
    float32 approximation the enumeration ranks with."""
    import filter_sk4_evolution as sk4

    circuit = sk4.SPEC.circuit_factory()
    sk4.SPEC.setup_hook(circuit, target)
    return float(sk4.SPEC.evaluator.fitness(circuit, sk4.resistor_mapper(taps), target))


def search(target: str) -> dict:
    import filter_baseline
    import filter_evaluation
    import filter_sk4_evolution as sk4

    pot = sk4.pot_10k
    evaluator = sk4.SPEC.evaluator

    drift = _check_separability(target)
    logger.info(f'separability: stage A moves {drift:.2e} dB when stage B changes')
    if drift > 1e-9:
        raise RuntimeError(f'stages are not separable ({drift:.3e} dB); the '
                           f'exhaustive decomposition would be invalid')

    pairs = list(itertools.product(range(1, pot.tap_points + 1), repeat=2))
    started = time.perf_counter()
    first_db, second_db, freqs = _simulate_all(target, pairs)
    logger.info(f'simulated both stages in {time.perf_counter() - started:.0f} s')

    floor = np.float32(filter_evaluation._DB_FLOOR)
    target_db = np.maximum(evaluator.target_db(freqs, target), floor).astype(np.float32)
    weights = evaluator.weights(freqs, target).astype(np.float32)
    weight_total = float(weights.sum())
    index_of = {pair: i for i, pair in enumerate(pairs)}

    def rmse_of(taps):
        error = np.maximum(first_db[index_of[(taps[0], taps[1])]]
                           + second_db[index_of[(taps[2], taps[3])]], floor) - target_db
        return float(np.sqrt(np.sum(weights * error ** 2) / weight_total))

    baseline_taps = filter_baseline.design(sk4, target)['taps']
    baseline_rmse = rmse_of(baseline_taps)

    best_rmse, best_pair, beat_baseline = np.float32(np.inf), None, 0
    started = time.perf_counter()
    for begin in range(0, len(pairs), _CHUNK):
        total = first_db[begin:begin + _CHUNK][:, None, :] + second_db[None, :, :]
        np.maximum(total, floor, out=total)
        total -= target_db
        total *= total
        rmse = np.sqrt(np.einsum('abf,f->ab', total, weights) / weight_total)

        beat_baseline += int((rmse < baseline_rmse).sum())
        flat = int(np.argmin(rmse))
        if rmse.flat[flat] < best_rmse:
            best_rmse = rmse.flat[flat]
            row, column = divmod(flat, rmse.shape[1])
            best_pair = (begin + row, column)
    logger.info(f'combined {len(pairs) ** 2:,} candidates in '
                f'{time.perf_counter() - started:.0f} s')

    optimum_taps = list(pairs[best_pair[0]]) + list(pairs[best_pair[1]])
    combinations = len(pairs) ** 2
    return {
        'target': target,
        'combinations': combinations,
        'optimum_taps': optimum_taps,
        'optimum_fitness': _exact_fitness(target, optimum_taps),
        'optimum_rmse_db': float(best_rmse),
        'baseline_taps': baseline_taps,
        'baseline_fitness': _exact_fitness(target, baseline_taps),
        'baseline_rmse_db': baseline_rmse,
        'baseline_rank': beat_baseline + 1,
        'baseline_percentile': 100.0 * (1 - beat_baseline / combinations),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', help='Run only for this target')
    parser.add_argument('--out', help='Write results to this JSON file')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    import filter_sk4_evolution as sk4
    targets = [args.target] if args.target else list(sk4.TARGETS)

    results = []
    for target in targets:
        logger.info(f'\n=== {target} ===')
        result = search(target)
        results.append(result)
        logger.info(f"otimo:    taps {result['optimum_taps']} "
                    f"fitness {result['optimum_fitness']:.6f} "
                    f"rmse {result['optimum_rmse_db']:.4f} dB")
        logger.info(f"baseline: taps {result['baseline_taps']} "
                    f"fitness {result['baseline_fitness']:.6f} "
                    f"rmse {result['baseline_rmse_db']:.4f} dB")
        logger.info(f"o projeto analitico e o {result['baseline_rank']}o melhor de "
                    f"{result['combinations']:,} combinacoes "
                    f"(percentil {result['baseline_percentile']:.4f})")

    if args.out:
        with open(args.out, 'w') as handle:
            json.dump(results, handle, indent=2)


if __name__ == '__main__':
    main()
