"""Statistics for the 2400-run campaign in `simulations/campaign/`.

`evaluate_sims.py` is shaped around the amplifier's time-domain columns and cannot
describe the filter, so the campaign's numbers are computed here. Everything this
prints is derived from the CSVs alone — no simulation — so any figure quoted in the
write-up can be regenerated and checked.

    python3 analyze_campaign.py

Two properties of the data drive the choices below:

  * 4R and 8R are *independent* samples, not paired. The same seed feeds a 4-gene
    and an 8-gene GA, whose initial populations are unrelated, so a per-seed
    difference is not a paired observation. Hence Mann-Whitney, not a signed test.
  * `cutoff_error_percent` in the CSV measures a Chebyshev's realized -3 dB point
    against its *nominal* cutoff, which for Chebyshev I is the ripple-band edge.
    Those differ by ~10% in the ideal filter, so the column overstates the error.
    `cutoff_accuracy` re-references it to the ideal response's own -3 dB point.
"""
import argparse
import collections
import csv
import os

import numpy as np
import scipy.signal
from scipy import stats

import filter_evaluation

from filter_targets import ORDER, TARGETS

CAMPAIGN_DIR = 'simulations/campaign'
CONTROL_DIR = 'simulations/control'
FAMILIES = ('butterworth', 'chebyshev')
CUTOFFS = (1000, 1500, 2000, 3000)
GAINS = (5, 10, 15, 20)


def load(path: str) -> list:
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))


def group(rows: list, key) -> dict:
    out = collections.defaultdict(list)
    for row in rows:
        out[key(row)].append(row)
    return out


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """P(b < a) - P(b > a): how often b beats a, on a -1..1 scale."""
    less = (b[:, None] < a[None, :]).sum()
    greater = (b[:, None] > a[None, :]).sum()
    return (less - greater) / (a.size * b.size)


def holm(pvalues: list) -> list:
    """Holm-Bonferroni adjusted p-values, in the order given."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def integrity(filter_rows: list, amp_rows: list) -> None:
    print('== Integrity ==')
    for name, rows, keys in (('filter', filter_rows, ('variant', 'target')),
                             ('amp', amp_rows, ('variant', 'desired_gain'))):
        cells = collections.Counter(tuple(r[k] for k in keys) for r in rows)
        seeds = collections.Counter((tuple(r[k] for k in keys), r['seed']) for r in rows)
        failures = sum(int(r['fitness_eval_failures']) for r in rows)
        generations = set(r['generations_completed'] for r in rows)
        print(f'  {name}: {len(rows)} rows, {len(cells)} cells, '
              f'{min(cells.values())}-{max(cells.values())} seeds per cell')
        print(f'    duplicate (cell, seed): {sum(1 for n in seeds.values() if n > 1)}'
              f'   eval failures: {failures}'
              f'   generations_completed: {sorted(generations)}')


def machine_consistency(directory: str) -> None:
    """The shards are disjoint halves of the same cells, so they must agree."""
    print('\n== Shard agreement ==')
    for label, pattern, column in (('filter', 'filter_%sd2.csv', 'rmse_db'),
                                   ('amp', 'amp_%sd2.csv', 'avg_error_percent')):
        paths = [os.path.join(directory, pattern % n) for n in (1, 2)]
        if not all(os.path.exists(p) for p in paths):
            print(f'  {label}: per-shard CSVs absent, skipped')
            continue
        first, second = (np.array([float(r[column]) for r in load(p)]) for p in paths)
        _, p = stats.mannwhitneyu(first, second, alternative='two-sided')
        print(f'  {label} {column}: shard medians {np.median(first):.5f} / '
              f'{np.median(second):.5f}, p = {p:.3f}')


def budget(filter_rows: list, amp_rows: list) -> None:
    """The 4R/8R comparison is confounded unless these budgets match."""
    print('\n== Search budget (CONFOUND if the variants differ) ==')
    for name, rows in (('filter', filter_rows), ('amp', amp_rows)):
        seen = sorted(set((r['variant'], int(r['population']), int(r['generations']))
                          for r in rows))
        for variant, population, generations in seen:
            print(f'  {name} {variant}: population {population} x {generations} '
                  f'generations = {population * generations} evaluations')
        if len(set(p * g for _, p, g in seen)) > 1:
            print(f'    -> {name}: budgets DIFFER between variants; a difference in '
                  'outcome cannot be attributed to resistor resolution alone')
    print('\n  Was the budget exhausted?')
    for variant in ('4R', '8R'):
        best = np.array([float(r['best_generation']) for r in filter_rows
                         if r['variant'] == variant])
        total = int(filter_rows[0]['generations'])
        print(f'    filter {variant}: best generation median {np.median(best):.0f}, '
              f'{(best > 0.875 * total).sum()}/{best.size} improved after '
              f'{int(0.875 * total)}, {(best >= total).sum()} at the limit')


def filter_quality(filter_rows: list) -> dict:
    print('\n== Filter RMSE (dB), 4R vs 8R ==')
    by = collections.defaultdict(list)
    for r in filter_rows:
        by[(r['target'], r['variant'])].append(float(r['rmse_db']))

    header = (f"{'target':<18}{'4R med':>9}{'4R best':>9}{'4R wrst':>9}"
              f"{'8R med':>9}{'8R best':>9}{'8R wrst':>9}{'ratio':>8}"
              f"{'p(Holm)':>11}{'delta':>8}")
    print(header)
    print('-' * len(header))
    pvalues, cells = [], []
    for family in FAMILIES:
        for cutoff in CUTOFFS:
            target = f'{family}_{cutoff}'
            a = np.array(sorted(by[(target, '4R')]))
            b = np.array(sorted(by[(target, '8R')]))
            _, p = stats.mannwhitneyu(b, a, alternative='two-sided')
            pvalues.append(p)
            cells.append((target, a, b))
    for (target, a, b), p in zip(cells, holm(pvalues)):
        print(f'{target:<18}{np.median(a):>9.4f}{a[0]:>9.4f}{a[-1]:>9.4f}'
              f'{np.median(b):>9.4f}{b[0]:>9.4f}{b[-1]:>9.4f}'
              f'{np.median(a) / np.median(b):>7.2f}x{p:>11.2e}'
              f'{cliffs_delta(a, b):>8.3f}')
    pooled_a = np.concatenate([a for _, a, _ in cells])
    pooled_b = np.concatenate([b for _, _, b in cells])
    _, p = stats.mannwhitneyu(pooled_b, pooled_a, alternative='less')
    print(f"\n  pooled {pooled_a.size} vs {pooled_b.size}: "
          f'median {np.median(pooled_a):.4f} vs {np.median(pooled_b):.4f} dB '
          f'({np.median(pooled_a) / np.median(pooled_b):.2f}x), one-sided p = {p:.2e}')
    return by


def versus_baseline(by: dict, baseline_path: str) -> None:
    if not os.path.exists(baseline_path):
        print('\n== GA vs analytical baseline ==\n  baseline.csv absent; run '
              'filter_baseline.py for both --circuit values')
        return
    base = {}
    for row in load(baseline_path):
        variant = '8R' if '8r' in row['ckt_name'] else '4R'
        base[(row['response'], variant)] = float(row['rmse_db'])

    print('\n== GA vs analytical baseline (same evaluator) ==')
    header = (f"{'target':<18}{'var':>4}{'baseline':>10}{'GA med':>9}"
              f"{'GA best':>9}{'ratio':>8}{'GA wins':>10}")
    print(header)
    print('-' * len(header))
    summary = collections.defaultdict(list)
    for family in FAMILIES:
        for cutoff in CUTOFFS:
            target = f'{family}_{cutoff}'
            for variant in ('4R', '8R'):
                ga = np.array(sorted(by[(target, variant)]))
                reference = base[(target, variant)]
                ratio = reference / np.median(ga)
                wins = int((ga < reference).sum())
                summary[variant].append((ratio, wins, ga.size))
                flag = '  <-- GA loses' if ratio < 1 else ''
                print(f'{target:<18}{variant:>4}{reference:>10.4f}'
                      f'{np.median(ga):>9.4f}{ga[0]:>9.4f}{ratio:>7.2f}x'
                      f'{wins:>7}/{ga.size}{flag}')
    print()
    for variant in ('4R', '8R'):
        ratios = np.array([r for r, _, _ in summary[variant]])
        wins = sum(w for _, w, _ in summary[variant])
        total = sum(n for _, _, n in summary[variant])
        print(f'  {variant}: ratio {ratios.min():.2f}x-{ratios.max():.2f}x, '
              f'geometric mean {np.exp(np.mean(np.log(ratios))):.2f}x, '
              f'beats baseline in {wins}/{total} runs')
    for variant in ('4R', '8R'):
        values = [base[(f'{f}_{c}', variant)] for f in FAMILIES for c in CUTOFFS]
        print(f'  baseline alone, {variant}: median {np.median(values):.4f} dB')


def ideal_minus_3db(target: str) -> float:
    """The -3 dB frequency of the ideal response, as the evaluator would measure it.

    Two things make this the right reference. First, for a Chebyshev I the nominal
    cutoff is the ripple-band edge, and the ideal response's own -3 dB point sits
    about 10% above it, so the nominal figure is not a reference at all.

    Second, it is measured on the *same* sweep grid and with the same log
    interpolation the evaluator uses for the realized crossing. That grid is coarse
    -- 20 points per decade -- and reading a -3 dB crossing off it is biased by up to
    0.8% depending on where the crossing falls between samples. Since both sides
    carry the same bias, the difference between them does not: an exactly-ideal
    filter reads as zero error. Pairing a coarse realized value with a dense-grid
    reference would instead report that bias as filter error.
    """
    spec = TARGETS[target]
    if spec['response'] == 'butterworth':
        b, a = scipy.signal.butter(ORDER, 2 * np.pi * spec['cutoff_hz'],
                                   btype='low', analog=True)
    else:
        b, a = scipy.signal.cheby1(ORDER, spec['ripple_db'], 2 * np.pi * spec['cutoff_hz'],
                                   btype='low', analog=True)
    freqs = _sweep_grid()
    _, h = scipy.signal.freqs(b, a, worN=2 * np.pi * freqs)
    db = 20 * np.log10(np.abs(h))
    db -= db[0]
    return filter_evaluation._minus_3db_crossing(freqs, db)


def _sweep_grid(start: float = 10.0, stop: float = 100_000.0,
                points_per_decade: int = 20) -> np.ndarray:
    """The AC sweep the evaluator runs, which the CSV's realized cutoff comes from."""
    decades = np.log10(stop / start)
    return start * 10 ** np.linspace(0, decades, int(decades * points_per_decade) + 1)


def cutoff_accuracy(filter_rows: list) -> None:
    print('\n== Cutoff accuracy vs the ideal response\'s own -3 dB point ==')
    by = group(filter_rows, lambda r: (r['target'], r['variant']))
    header = (f"{'target':<18}{'nominal':>9}{'ideal -3dB':>12}"
              f"{'4R med%':>9}{'4R max%':>9}{'8R med%':>9}{'8R max%':>9}")
    print(header)
    print('-' * len(header))
    worst = collections.defaultdict(list)
    for family in FAMILIES:
        for cutoff in CUTOFFS:
            target = f'{family}_{cutoff}'
            reference = ideal_minus_3db(target)
            errors = {}
            for variant in ('4R', '8R'):
                e = np.array([abs(float(r['cutoff_realized_hz']) - reference)
                              / reference * 100 for r in by[(target, variant)]])
                errors[variant] = e
                worst[variant].append(e.max())
            print(f'{target:<18}{cutoff:>9}{reference:>12.1f}'
                  f"{np.median(errors['4R']):>9.3f}{errors['4R'].max():>9.3f}"
                  f"{np.median(errors['8R']):>9.3f}{errors['8R'].max():>9.3f}")
    print()
    for variant in ('4R', '8R'):
        print(f'  {variant}: worst cutoff error over every run = {max(worst[variant]):.3f}%')


def amplifier(amp_rows: list) -> None:
    print('\n== Amplifier: a floor plus a trap, not a spread ==')
    by = collections.defaultdict(list)
    for r in amp_rows:
        by[(int(float(r['desired_gain'])), r['variant'])].append(r)

    header = (f"{'gain':>5}{'var':>5}{'med%':>9}{'best%':>9}"
              f"{'<=1.05x best':>14}{'>=2x best':>12}{'p(Holm)':>11}")
    print(header)
    print('-' * len(header))
    pvalues, cells = [], []
    for gain in GAINS:
        a = np.array(sorted(float(r['avg_error_percent']) for r in by[(gain, '4R')]))
        b = np.array(sorted(float(r['avg_error_percent']) for r in by[(gain, '8R')]))
        _, p = stats.mannwhitneyu(b, a, alternative='two-sided')
        pvalues.append(p)
        cells.append((gain, a, b))
    adjusted = holm(pvalues)
    for (gain, a, b), p in zip(cells, adjusted):
        for variant, values in (('4R', a), ('8R', b)):
            best = values[0]
            print(f'{gain:>5}{variant:>5}{np.median(values):>9.4f}{best:>9.4f}'
                  f'{(values <= 1.05 * best).sum():>11}/{values.size}'
                  f'{(values >= 2 * best).sum():>9}/{values.size}'
                  f'{p:>11.2e}')

    print('\n  Output swing of near-optimal vs trapped runs (CSV holds 100x volts,'
          ' so the ideal is +/-gain):')
    for gain in GAINS:
        for variant in ('4R', '8R'):
            rows = by[(gain, variant)]
            threshold = 2 * min(float(r['avg_error_percent']) for r in rows)
            for label, group_rows in (
                    ('near-opt', [r for r in rows if float(r['avg_error_percent']) < threshold]),
                    ('trapped', [r for r in rows if float(r['avg_error_percent']) >= threshold])):
                if not group_rows:
                    continue
                high = np.median([float(r['max_voltage']) for r in group_rows])
                low = np.median([float(r['min_voltage']) for r in group_rows])
                best_gen = np.median([float(r['best_generation']) for r in group_rows])
                print(f'    gain {gain:>2} {variant} {label:<9} n={len(group_rows):>3}  '
                      f'Vmax {high:>7.3f}  Vmin {low:>8.3f}  '
                      f'swing {high - low:>7.3f}  best gen {best_gen:>4.0f}')


def _cell_order(cell: str):
    """Sort filter cells by family then cutoff, and amp cells numerically by gain."""
    family, _, cutoff = cell.rpartition('_')
    if family and cutoff.isdigit():
        return (family, int(cutoff))
    return ('', float(cell)) if cell.replace('.', '', 1).isdigit() else (cell, 0)


def matched_budget(campaign_dir: str, control_dir: str) -> None:
    """4R re-run at 8R's population, against 8R -- the confound `budget` reports.

    Until this comparison exists, a 4R-vs-8R difference mixes resistor resolution
    with the number of evaluations each variant was allowed. With both at the same
    population, whatever remains is resolution.
    """
    print('\n== Matched budget: 4R re-run at 8R\'s population ==')
    pairs = (('filter', 'filter_all.csv', 'filter_control.csv', 'rmse_db', 'target'),
             ('amp', 'amp_all.csv', 'amp_control.csv', 'avg_error_percent', 'desired_gain'))
    for family, original, control, column, key in pairs:
        original_path = os.path.join(campaign_dir, original)
        control_path = os.path.join(control_dir, control)
        if not (os.path.exists(original_path) and os.path.exists(control_path)):
            print(f'  {family}: needs {original} and {control}; skipped')
            continue

        campaign_rows = load(original_path)
        control_rows = load(control_path)
        populations = set(r['population'] for r in control_rows)
        variants = set(r['variant'] for r in control_rows)
        print(f'\n  {family}: control is variant {sorted(variants)} at population '
              f'{sorted(populations)}, {len(control_rows)} runs')

        low, high, control_by = {}, {}, collections.defaultdict(list)
        for row in campaign_rows:
            (low if row['variant'] == '4R' else high).setdefault(row[key], []).append(
                float(row[column]))
        for row in control_rows:
            control_by[row[key]].append(float(row[column]))

        header = (f"    {'cell':<18}{'4R base':>10}{'4R matched':>12}{'8R':>10}"
                  f"{'budget':>9}{'resolution':>12}{'p(Holm)':>10}{'delta':>8}")
        print(header)
        print('    ' + '-' * (len(header) - 4))
        cells, pvalues = [], []
        for cell in sorted(control_by, key=_cell_order):
            matched = np.array(control_by[cell])
            better = np.array(high[cell])
            _, p = stats.mannwhitneyu(better, matched, alternative='two-sided')
            pvalues.append(p)
            cells.append((cell, np.array(low[cell]), matched, better))
        for (cell, base, matched, better), p in zip(cells, holm(pvalues)):
            print(f'    {cell:<18}{np.median(base):>10.4f}{np.median(matched):>12.4f}'
                  f'{np.median(better):>10.4f}'
                  f'{np.median(base) / np.median(matched):>8.2f}x'
                  f'{np.median(matched) / np.median(better):>11.2f}x'
                  f'{p:>10.2e}{cliffs_delta(matched, better):>8.3f}')

        pooled_base = np.concatenate([b for _, b, _, _ in cells])
        pooled_matched = np.concatenate([m for _, _, m, _ in cells])
        pooled_better = np.concatenate([h for _, _, _, h in cells])
        _, p = stats.mannwhitneyu(pooled_better, pooled_matched, alternative='less')
        print(f'\n    pooled: 4R base {np.median(pooled_base):.4f} | '
              f'4R matched {np.median(pooled_matched):.4f} | 8R {np.median(pooled_better):.4f}')
        print(f'      the extra budget alone is worth '
              f'{np.median(pooled_base) / np.median(pooled_matched):.2f}x')
        print(f'      resolution at matched budget is worth '
              f'{np.median(pooled_matched) / np.median(pooled_better):.2f}x, '
              f'one-sided p = {p:.2e}')
        print(f'      unmatched, the two together read as '
              f'{np.median(pooled_base) / np.median(pooled_better):.2f}x')


def consistency_versus_baseline(campaign_dir: str, control_dir: str) -> None:
    """Which of the two designs is erratic across cells -- the GA or the analytical one.

    A per-cell median comparison answers "who wins here", which turns out to be the
    wrong question: the analytical design's quality swings by nearly an order of
    magnitude depending on where the ideal resistors happen to land relative to the
    tap grid, while the GA's varies far less. The spread is the result.
    """
    baseline_path = os.path.join(campaign_dir, 'baseline.csv')
    control_path = os.path.join(control_dir, 'filter_control.csv')
    if not (os.path.exists(baseline_path) and os.path.exists(control_path)):
        print('\n== Spread across cells ==\n  baseline.csv or filter_control.csv absent')
        return

    baseline = {}
    for row in load(baseline_path):
        if '8r' not in row['ckt_name']:
            baseline[row['response']] = float(row['rmse_db'])
    control = collections.defaultdict(list)
    for row in load(control_path):
        control[row['target']].append(float(row['rmse_db']))

    targets = [f'{f}_{c}' for f in FAMILIES for c in CUTOFFS]
    analytical = np.array([baseline[t] for t in targets])
    medians = np.array([np.median(control[t]) for t in targets])
    bests = np.array([min(control[t]) for t in targets])

    print('\n== Spread across cells: who is erratic, the GA or the analytical design ==')
    for label, values in (('analytical design', analytical),
                          ('GA median', medians), ('GA best', bests)):
        print(f'  {label:<20} {values.min():.4f} to {values.max():.4f} dB '
              f'({values.max() / values.min():.1f}x spread, CV '
              f'{values.std(ddof=1) / values.mean():.2f})')
    print(f'  GA best beats the analytical design in {(bests < analytical).sum()}/'
          f'{len(targets)} cells, GA median in {(medians < analytical).sum()}/{len(targets)}')
    # If the GA wins exactly where the analytical design lands badly, its apparent
    # advantage is a property of the baseline, not of the GA.
    correlation = np.corrcoef(analytical, analytical / medians)[0, 1]
    print(f'  correlation between the analytical error and the GA\'s advantage over it: '
          f'r = {correlation:+.3f}')


def cost(filter_rows: list, amp_rows: list) -> None:
    print('\n== Cost ==')
    rows = filter_rows + amp_rows
    total = sum(float(r['runtime_s']) for r in rows)
    print(f'  {len(rows)} runs, {total / 3600:.1f} h of serial CPU time')
    for name, subset in (('filter', filter_rows), ('amp', amp_rows)):
        for variant in ('4R', '8R'):
            times = np.array([float(r['runtime_s']) for r in subset
                              if r['variant'] == variant])
            print(f'    {name} {variant}: median {np.median(times):>6.1f} s per run')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dir', default=CAMPAIGN_DIR,
                        help='directory holding filter_all.csv, amp_all.csv, baseline.csv')
    parser.add_argument('--control-dir', default=CONTROL_DIR,
                        help='directory holding the matched-budget control CSVs')
    args = parser.parse_args()

    filter_rows = load(os.path.join(args.dir, 'filter_all.csv'))
    amp_rows = load(os.path.join(args.dir, 'amp_all.csv'))

    integrity(filter_rows, amp_rows)
    machine_consistency(args.dir)
    budget(filter_rows, amp_rows)
    by = filter_quality(filter_rows)
    versus_baseline(by, os.path.join(args.dir, 'baseline.csv'))
    cutoff_accuracy(filter_rows)
    amplifier(amp_rows)
    matched_budget(args.dir, args.control_dir)
    consistency_versus_baseline(args.dir, args.control_dir)
    cost(filter_rows, amp_rows)


if __name__ == '__main__':
    main()
