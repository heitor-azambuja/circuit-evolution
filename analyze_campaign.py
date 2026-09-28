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

    python3 analyze_campaign.py --paper

prints instead every number the ACDSA 2027 paper quotes, section by section and in
its order. The paper reports both variants at population 40, so there 4R comes from
the matched re-run in the control directory and 8R from the campaign. A few of its
numbers describe circuits rather than runs (the analytical designs, the op-amp
model, the R1/R2 swaps), and `--paper` re-simulates those, so it needs ngspice.
"""
import argparse
import collections
import csv
import json
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


def median_best_of_n(values, n: int) -> float:
    """Median of the best of `n` runs drawn without replacement -- computed, not sampled.

    For sorted observations v[0] <= ... <= v[N-1], a draw of size n has its minimum at
    or above v[i] exactly when all n come from the N-i values at or after i, so

        P(min >= v[i]) = C(N-i, n) / C(N, n)

    and the median is the first v[i] whose probability of being met or beaten reaches
    one half. Exact, and faster than resampling by orders of magnitude.
    """
    from math import comb

    ordered = np.sort(np.asarray(values, dtype=float))
    total = ordered.size
    if n >= total:
        return float(ordered[0])
    denominator = comb(total, n)
    for i in range(total):
        # P(min <= v[i]) = 1 - P(all n drawn from strictly after i)
        if 1.0 - comb(total - i - 1, n) / denominator >= 0.5:
            return float(ordered[i])
    return float(ordered[-1])


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


# -- The paper -----------------------------------------------------------------------
#
# Everything below backs `--paper`. The paper reports both variants at population 40,
# one search budget; the campaign's 4R at population 20 and the 800-generation side
# test are not part of it. Circuit imports stay inside the functions so the default,
# CSV-only run never needs ngspice.

OPTIMA_PATH = 'simulations/optima_4r.json'
PAPER_TARGETS = [f'{f}_{c}' for f in FAMILIES for c in CUTOFFS]


def _paper_runs(campaign_dir: str, control_dir: str) -> tuple:
    """Rows keyed by (target, variant) for the filter and (gain, variant) for the amp."""
    filters, amps = collections.defaultdict(list), collections.defaultdict(list)
    for row in load(os.path.join(control_dir, 'filter_control.csv')):
        filters[(row['target'], '4R')].append(row)
    for row in load(os.path.join(campaign_dir, 'filter_all.csv')):
        if row['variant'] == '8R':
            filters[(row['target'], '8R')].append(row)
    for row in load(os.path.join(control_dir, 'amp_control.csv')):
        amps[(int(float(row['desired_gain'])), '4R')].append(row)
    for row in load(os.path.join(campaign_dir, 'amp_all.csv')):
        if row['variant'] == '8R':
            amps[(int(float(row['desired_gain'])), '8R')].append(row)
    return filters, amps


def _values(rows: list, column: str) -> np.ndarray:
    return np.array([float(row[column]) for row in rows])


def _filter_baselines(campaign_dir: str) -> dict:
    out = {}
    for row in load(os.path.join(campaign_dir, 'baseline.csv')):
        variant = '8R' if '8r' in row['ckt_name'] else '4R'
        out[(row['response'], variant)] = float(row['rmse_db'])
    return out


def _filter_metrics(target: str, resistances: list, op_amp) -> dict:
    """One AC sweep of the cascade with these resistances; op_amp None is an ideal buffer."""
    import circuits
    import filter_sk4_evolution
    import filter_targets

    circuit = circuits.SallenKeyLowPass(op_amp=op_amp)
    filter_targets.configure(circuit, target)
    return filter_sk4_evolution.SPEC.evaluator.metrics(circuit, resistances, target)


def _ideal_resistors(family: str, cutoff_hz: float) -> list:
    """The analytical design's unrounded resistors for any cutoff, in the circuit's order."""
    import filter_design
    import filter_targets

    spec = filter_targets.RESPONSES[family]
    caps = spec['capacitors_nf']
    stages = filter_design.stage_parameters(ORDER, spec['response'], cutoff_hz,
                                            ripple_db=spec['ripple_db'])
    resistors = []
    for (wo, q), (c1, c2) in zip(stages, ((caps[0], caps[1]), (caps[2], caps[3]))):
        resistors.extend(filter_design.ideal_resistors(wo, q, c1 * 1e-9, c2 * 1e-9))
    return resistors


def paper_circuits() -> None:
    """Section II: why the capacitors differ, and what limits the filter's accuracy."""
    import re

    import amp_design
    import circuits
    import filter_baseline
    import filter_design
    import filter_sk4_evolution as sk4
    import filter_sk8_evolution as sk8
    import filter_targets

    print('\n== Paper II: circuits and components ==')
    for family, spec in filter_targets.RESPONSES.items():
        stages = filter_design.stage_parameters(ORDER, spec['response'], 1000.0,
                                                ripple_db=spec['ripple_db'])
        print(f'  {family}: stage Q ' + ' and '.join(f'{q:.3f}' for _, q in stages))

    full_scale, taps = sk4.POT_SPECS[0]
    step = full_scale / (taps - 1)
    for family in filter_targets.RESPONSES:
        # The resistors scale as 1/fc, so the largest reaches the part's full scale at:
        lowest = 1000.0 * max(_ideal_resistors(family, 1000.0)) / full_scale
        smallest = min(_ideal_resistors(family, 3000.0))
        print(f'  {family}: the largest resistor reaches {full_scale / 1000:g} kOhm at '
              f'{lowest:.0f} Hz; at 3 kHz the smallest is {smallest:.0f} Ohm, where one '
              f'{step:.0f} Ohm step is {step / smallest * 100:.1f}% of it')

    target = 'butterworth_2000'
    designed = filter_baseline.design(sk4, target)['taps']
    swapped = [designed[1], designed[0], designed[3], designed[2]]
    for op_amp, label in ((circuits.MCP6002, 'MCP6002'), (None, 'ideal buffer')):
        before, after = (_filter_metrics(target, sk4.resistor_mapper(t), op_amp)['rmse_db']
                         for t in (designed, swapped))
        print(f'  {target}, R1/R2 swapped in both stages, {label}: '
              f'{before:.4f} -> {after:.4f} dB')

    for target in ('butterworth_1000', 'chebyshev_1000'):
        metrics = _filter_metrics(target, filter_baseline.design(sk4, target)['ideal_resistors'],
                                  None)
        deviation = np.abs(metrics['response_db'] - metrics['target_db']).max()
        print(f'  {target}, ideal op-amp and unquantized resistors: largest deviation '
              f'{deviation:.1e} dB over {metrics["freqs"].size} points')

    for target in ('butterworth_1000', 'chebyshev_1000'):
        d4, d8 = filter_baseline.design(sk4, target), filter_baseline.design(sk8, target)
        parts = (('rounding alone, 4R', d4['quantized_resistors'], None),
                 ('rounding alone, 8R', d8['quantized_resistors'], None),
                 ('op-amp alone', d4['ideal_resistors'], circuits.MCP6002),
                 ('both, 4R', d4['quantized_resistors'], circuits.MCP6002),
                 ('both, 8R', d8['quantized_resistors'], circuits.MCP6002))
        print(f'  {target}: ' + ', '.join(
            f'{label} {_filter_metrics(target, r, op_amp)["rmse_db"]:.4f}'
            for label, r, op_amp in parts) + ' dB')

    # The textbook divider depends on the transistor's current gain; take the model's.
    with open('components.spice') as handle:
        beta = float(re.search(r'\.model\s+bc547b\b.*?\bBF=([\d.]+)', handle.read(),
                               re.I | re.S).group(1))
    divider = amp_design.textbook_divider_total(amp_design.design(10)['collector_current'],
                                                beta)
    print(f'  amplifier: Rc at gain 5 is {amp_design.design(5)["resistances"][2] / 1000:.1f} '
          f'kOhm; the textbook divider at gain 10 (BF = {beta:g}) needs '
          f'{divider / 1e6:.2f} MOhm')


def paper_objective(control_dir: str, optima: dict) -> None:
    """Section III: the DC normalization, the cutoff reference, the enumeration check."""
    from scipy import optimize

    import filter_baseline
    import filter_sk4_evolution as sk4

    print('\n== Paper III: evolutionary method ==')
    evaluator = sk4.SPEC.evaluator
    target = 'chebyshev_1000'
    metrics = _filter_metrics(target,
                              filter_baseline.design(sk4, target)['quantized_resistors'], None)
    unnormalized = metrics['target_db'] + evaluator._dc_gain_db[target]
    floor = filter_evaluation._DB_FLOOR
    error = np.maximum(metrics['response_db'], floor) - np.maximum(unnormalized, floor)
    weights = evaluator.weights(metrics['freqs'], target)
    print(f'  {target}, 4R analytical design, ideal op-amp: {metrics["rmse_db"]:.4f} dB '
          f'against the normalized target, '
          f'{np.sqrt(np.sum(weights * error ** 2) / np.sum(weights)):.4f} dB against the raw one')

    grid, worst = _sweep_grid(), 0.0
    for target in PAPER_TARGETS:
        cutoff = TARGETS[target]['cutoff_hz']
        exact = optimize.brentq(lambda f: evaluator.target_db([f], target)[0] + 3.0,
                                0.5 * cutoff, 2.0 * cutoff, xtol=1e-9)
        on_grid = filter_evaluation._minus_3db_crossing(grid, evaluator.target_db(grid, target))
        worst = max(worst, abs(on_grid - exact) / exact * 100)
        if target == 'chebyshev_1000':
            offset = (exact - cutoff) / cutoff * 100
    print(f'  0.5 dB Chebyshev, normalized: -3 dB point {offset:+.1f}% from the nominal cutoff')
    print(f'  sweep grid: a single -3 dB crossing is biased by up to {worst:.2f}%')

    differences = []
    for row in load(os.path.join(control_dir, 'filter_control.csv')):
        optimum = optima[row['target']]
        if json.loads(row['solution']) == optimum['optimum_taps']:
            differences.append(abs(float(row['rmse_db']) - optimum['optimum_rmse_db']))
    print(f'  enumeration vs full netlist: {len(differences)} runs end on the optimal taps; '
          f'the two scores differ by at most {max(differences):.1e} dB')


def paper_design(filters: dict, amps: dict) -> None:
    """Section IV: what the runs are."""
    print('\n== Paper IV: experimental design ==')
    arms = list(filters.values()) + list(amps.values())
    rows = [row for arm in arms for row in arm]
    failures = sum(int(float(row.get('fitness_eval_failures') or 0)) for row in rows)
    print(f'  {len(rows)} runs in {len(arms)} arms of {sorted(set(len(a) for a in arms))} runs; '
          f'population {sorted(set(r["population"] for r in rows))}, generations '
          f'{sorted(set(r["generations_completed"] for r in rows))}, '
          f'{failures} failed evaluations')


def paper_resolution(filters: dict, base: dict) -> None:
    """Section V-A: what the second potentiometer buys at equal budget."""
    print('\n== Paper V-A: resistor resolution ==')
    low = {t: _values(filters[(t, '4R')], 'rmse_db') for t in PAPER_TARGETS}
    high = {t: _values(filters[(t, '8R')], 'rmse_db') for t in PAPER_TARGETS}
    pooled_low = np.concatenate(list(low.values()))
    pooled_high = np.concatenate(list(high.values()))
    _, p = stats.mannwhitneyu(pooled_high, pooled_low, alternative='less')
    print(f'  pooled medians: 4R {np.median(pooled_low):.4f} dB, 8R {np.median(pooled_high):.4f} '
          f'dB, a factor of {np.median(pooled_low) / np.median(pooled_high):.2f} '
          f'(one-sided p = {p:.1e})')
    pvalues = [stats.mannwhitneyu(high[t], low[t], alternative='two-sided')[1]
               for t in PAPER_TARGETS]
    deltas = [cliffs_delta(low[t], high[t]) for t in PAPER_TARGETS]
    lower = sum(np.median(high[t]) < np.median(low[t]) for t in PAPER_TARGETS)
    print(f'  8R lower in {lower}/8; largest p(Holm) {max(holm(pvalues)):.1e}; '
          f'Cliff delta {min(deltas):.2f} to {max(deltas):.2f}')

    for variant in ('4R', '8R'):
        medians, worst = [], 0.0
        for target in PAPER_TARGETS:
            reference = ideal_minus_3db(target)
            errors = np.abs(_values(filters[(target, variant)], 'cutoff_realized_hz')
                            - reference) / reference * 100
            medians.append(np.median(errors))
            worst = max(worst, errors.max())
        print(f'  -3 dB point, {variant}: medians {min(medians):.2f}% to {max(medians):.2f}%, '
              f'worst {worst:.2f}%')
    for variant in ('4R', '8R'):
        print(f'  analytical design, {variant}: median over specifications '
              f'{np.median([base[(t, variant)] for t in PAPER_TARGETS]):.4f} dB')


def paper_figure2(filters: dict) -> None:
    """Fig. 2: the analytical design and the median run of each variant, 3 kHz Chebyshev."""
    import circuits
    import filter_baseline
    import filter_sk4_evolution as sk4
    import filter_sk8_evolution as sk8

    target = 'chebyshev_3000'
    print(f'\n== Paper V-A: Fig. 2, {target} ==')

    def median_run(variant):
        # The lower of the two middle runs, the same one plot_paper draws.
        rows = sorted(filters[(target, variant)], key=lambda r: float(r['rmse_db']))
        return json.loads(rows[(len(rows) - 1) // 2]['solution'])

    curves = (('analytical design, 4R',
               sk4.resistor_mapper(filter_baseline.design(sk4, target)['taps'])),
              ('median run, 4R', sk4.resistor_mapper(median_run('4R'))),
              ('median run, 8R', sk8.resistor_mapper(median_run('8R'))))
    for label, resistances in curves:
        metrics = _filter_metrics(target, resistances, circuits.MCP6002)
        deviation = np.abs(metrics['response_db'] - metrics['target_db'])
        print(f'  {label}: {metrics["rmse_db"]:.4f} dB, largest deviation '
              f'{deviation.max():.2f} dB at {metrics["freqs"][deviation.argmax()] / 1000:.2f} kHz')


def paper_spread(filters: dict, base: dict) -> None:
    """Section V-B: which of the two designs varies with the specification."""
    print('\n== Paper V-B: consistency across specifications ==')
    analytical = np.array([base[(t, '4R')] for t in PAPER_TARGETS])
    runs = [_values(filters[(t, '4R')], 'rmse_db') for t in PAPER_TARGETS]
    medians = np.array([np.median(r) for r in runs])
    bests = np.array([r.min() for r in runs])
    wins = medians < analytical
    worst = int(np.argmax(medians / analytical))
    print(f'  median 4R run beats the analytical design in {wins.sum()}/8; largest loss '
          f'{PAPER_TARGETS[worst]}, {medians[worst]:.4f} against {analytical[worst]:.4f} dB '
          f'({medians[worst] / analytical[worst]:.2f}x)')

    runs8 = [_values(filters[(t, '8R')], 'rmse_db') for t in PAPER_TARGETS]
    analytical8 = np.array([base[(t, '8R')] for t in PAPER_TARGETS])
    ratios8 = analytical8 / np.array([np.median(r) for r in runs8])
    beating = sum(int((r < a).sum()) for r, a in zip(runs8, analytical8))
    print(f'  8R: the median run wins {int((ratios8 > 1).sum())}/8, geometric-mean factor '
          f'{np.exp(np.mean(np.log(ratios8))):.2f}; {beating}/{sum(r.size for r in runs8)} '
          f'runs win')

    for label, values in (('analytical design, 4R', analytical), ('median run', medians),
                          ('best run', bests)):
        print(f'  {label}: {values.min():.4f} to {values.max():.4f} dB, '
              f'{values.max() / values.min():.1f}-fold, CV {values.std(ddof=1) / values.mean():.2f}')
    rho, p = stats.spearmanr(analytical, medians)
    print(f'  median run vs analytical design across specifications: Spearman rho = '
          f'{rho:.2f}, p = {p:.2f}')
    print(f'  median analytical error where the median run loses {np.median(analytical[~wins]):.4f} '
          f'dB, where it wins {np.median(analytical[wins]):.4f} dB')


def paper_optimum(filters: dict, base: dict, optima: dict) -> None:
    """Section V-C: the GA and the analytical design against the enumerated optimum."""
    print('\n== Paper V-C: the global optimum ==')
    exact = 0
    for target in PAPER_TARGETS:
        best = min(filters[(target, '4R')], key=lambda r: float(r['rmse_db']))
        taps = json.loads(best['solution'])
        if taps == optima[target]['optimum_taps']:
            exact += 1
        else:
            print(f'  {target}: best run {taps} at {float(best["rmse_db"]):.6f} dB, optimum '
                  f'{optima[target]["optimum_taps"]} at {optima[target]["optimum_rmse_db"]:.6f} dB')
    print(f'  the best of 100 runs is the exact optimum in {exact}/8')

    optimum = np.array([optima[t]['optimum_rmse_db'] for t in PAPER_TARGETS])
    analytical = np.array([base[(t, '4R')] for t in PAPER_TARGETS])
    medians = np.array([np.median(_values(filters[(t, '4R')], 'rmse_db')) for t in PAPER_TARGETS])
    ranks = np.array([optima[t]['baseline_rank'] for t in PAPER_TARGETS])
    percentiles = [optima[t]['baseline_percentile'] for t in PAPER_TARGETS]
    print(f'  median run: {(medians / optimum).min():.1f} to {(medians / optimum).max():.1f} '
          f'times the optimum')
    ratio = analytical / optimum
    print(f'  analytical design: rank {ranks.min()} to {ranks.max()}, lowest percentile '
          f'{min(percentiles):.3f}, {ratio.min():.1f} to {ratio.max():.1f} times the optimum')
    loses = medians >= analytical
    print(f'  where the median run loses: ranks {sorted(ranks[loses].tolist())}, '
          f'{ratio[loses].min():.1f} to {ratio[loses].max():.1f} times the optimum; where it '
          f'wins: {ratio[~loses].min():.1f} to {ratio[~loses].max():.1f} times')


def paper_reassignment(optima: dict) -> None:
    """Section V-C: how much of the analytical design's error is its R1/R2 assignment."""
    import itertools

    import circuits
    import filter_sk4_evolution as sk4

    print('\n== Paper V-C: reassigning R1 and R2 in the analytical design ==')
    gains, exact = [], []
    for target in PAPER_TARGETS:
        taps = optima[target]['baseline_taps']
        candidates = []
        for swap_a, swap_b in itertools.product((False, True), repeat=2):
            order = (taps[1::-1] if swap_a else taps[:2]) + (taps[3:1:-1] if swap_b else taps[2:])
            rmse = _filter_metrics(target, sk4.resistor_mapper(order), circuits.MCP6002)['rmse_db']
            candidates.append((rmse, order))
        # Against the design as written, simulated the same way: the enumeration's own
        # score for it differs from a full-netlist one at the 1e-7 dB level.
        as_written = candidates[0][0]
        rmse, order = min(candidates)
        gains.append(as_written / rmse)
        if order == optima[target]['optimum_taps']:
            exact.append(target)
    gains = np.array(gains)
    print(f'  improves {int((gains > 1 + 1e-6).sum())}/8, median factor {np.median(gains):.2f}, '
          f'largest {gains.max():.2f}; the exact optimum for {", ".join(exact)}')


def paper_best_of_n(filters: dict, base: dict, optima: dict) -> None:
    """Section V-D and Fig. 4: keeping the best of N independent runs."""
    print('\n== Paper V-D: number of runs ==')
    runs = {t: _values(filters[(t, '4R')], 'rmse_db') for t in PAPER_TARGETS}
    within = sum(int((runs[t] <= 1.01 * optima[t]['optimum_rmse_db']).sum())
                 for t in PAPER_TARGETS)
    total = sum(r.size for r in runs.values())
    print(f'  {within}/{total} runs ({within / total * 100:.1f}%) within 1% of the optimum')
    for n in (1, 2, 3, 5, 10, 20):
        best = {t: median_best_of_n(runs[t], n) for t in PAPER_TARGETS}
        wins = sum(best[t] < base[(t, '4R')] for t in PAPER_TARGETS)
        ratios = [best[t] / optima[t]['optimum_rmse_db'] for t in PAPER_TARGETS]
        print(f'  N = {n:>2}: beats the analytical design in {wins}/8; {min(ratios):.2f} to '
              f'{max(ratios):.2f} times the optimum, median {np.median(ratios):.2f}')


def paper_amplifier(amps: dict) -> None:
    """Section V-E: the amplifier, whose analytical design is not exact."""
    import amp_design
    import campaign
    import components

    print('\n== Paper V-E: the amplifier ==')

    def analytical(variant, gain, quantized=True):
        module = campaign._module('amp', variant)
        spec = module.SPEC
        circuit = spec.circuit_factory()
        spec.setup_hook(circuit, gain)
        if quantized:
            resistances = spec.resistor_mapper(campaign.baseline_taps('amp', variant, gain))
        else:
            design = amp_design.design(gain)
            pot = components.DigitalPot(*module.POT_SPECS[0])
            r1, r2 = amp_design.divider(design['base_voltage'], pot)
            resistances = [r1, r2, design['resistances'][2], design['resistances'][3]]
        return spec.evaluator.metrics(circuit, resistances, gain)['avg_error_percent']

    base = {(gain, variant): analytical(variant, gain)
            for gain in GAINS for variant in ('4R', '8R')}
    for gain in GAINS:
        ideal = analytical('4R', gain, quantized=False)
        rounded = base[(gain, '4R')]
        print(f'  gain {gain}: analytical design {ideal:.2f}% unquantized, {rounded:.2f}% on 4R '
              f'taps (quantization {(rounded - ideal) / rounded * 100:+.1f}% of it), '
              f'{base[(gain, "8R")]:.2f}% on 8R taps')

    advantages, beaten, total = [], 0, 0
    for (gain, variant), rows in sorted(amps.items()):
        errors = _values(rows, 'avg_error_percent')
        beaten += int((errors < base[(gain, variant)]).sum())
        total += errors.size
        advantages.append(base[(gain, variant)] / np.median(errors))
    print(f'  {beaten}/{total} runs beat their variant\'s analytical design; the median run by '
          f'{min(advantages):.2f} to {max(advantages):.2f} times')

    distances = []
    for gain in GAINS:
        best = min(amps[(gain, '4R')], key=lambda r: float(r['avg_error_percent']))
        designed = campaign.baseline_taps('amp', '4R', gain)
        distances.append(max(abs(a - b) for a, b in zip(json.loads(best['solution']), designed)))
    print(f'  best 4R run vs the analytical taps: largest per-resistor distance '
          f'{min(distances)} to {max(distances)} taps')

    floors, stalled_medians, gaps, lower_4r = [], [], [], []
    for gain in GAINS:
        errors = {v: _values(amps[(gain, v)], 'avg_error_percent') for v in ('4R', '8R')}
        floors.append(min(e.min() for e in errors.values()))
        # Stalled: at least twice the best run of its own arm.
        stalled = np.concatenate([e[e >= 2 * e.min()] for e in errors.values()])
        stalled_medians.append(np.median(stalled))
        b4, b8 = errors['4R'].min(), errors['8R'].min()
        gaps.append(abs(b4 - b8) / min(b4, b8) * 100)
        if b4 < b8:
            lower_4r.append(gain)
    print('  floor (best run of each gain): ' + ', '.join(f'{f:.2f}%' for f in floors))
    print(f'  stalled runs: medians {min(stalled_medians):.1f}% to {max(stalled_medians):.1f}% '
          f'across gains')
    for variant in ('4R', '8R'):
        rows = amps[(5, variant)]
        errors = _values(rows, 'avg_error_percent')
        stalled = errors >= 2 * errors.min()
        # The CSV holds 100x volts, so x10 gives millivolts.
        high, low = _values(rows, 'max_voltage') * 10, _values(rows, 'min_voltage') * 10
        print(f'  gain 5, {variant}: stalled runs {np.median(high[stalled]):+.1f}/'
              f'{np.median(low[stalled]):.1f} mV, runs at the floor '
              f'{np.median(high[~stalled]):+.1f}/{np.median(low[~stalled]):.1f} mV')
    print(f'  best 4R vs best 8R run: at most {max(gaps):.2f}% apart; 4R lower at gains '
          f'{lower_4r}')

    pooled = {v: np.concatenate([_values(amps[(g, v)], 'avg_error_percent') for g in GAINS])
              for v in ('4R', '8R')}
    _, p = stats.mannwhitneyu(pooled['8R'], pooled['4R'], alternative='less')
    print(f'  pooled medians: 4R {np.median(pooled["4R"]):.4f}%, 8R {np.median(pooled["8R"]):.4f}%, '
          f'a factor of {np.median(pooled["4R"]) / np.median(pooled["8R"]):.2f} '
          f'(one-sided p = {p:.1e})')
    for variant in ('4R', '8R'):
        counts = [int((e <= 1.05 * e.min()).sum()) for e in
                  (_values(amps[(g, variant)], 'avg_error_percent') for g in GAINS)]
        print(f'  {variant}: runs within 1.05x of the best run of the gain, gains {list(GAINS)}: '
              f'{counts}')


def paper(campaign_dir: str, control_dir: str) -> None:
    filters, amps = _paper_runs(campaign_dir, control_dir)
    base = _filter_baselines(campaign_dir)
    with open(OPTIMA_PATH) as handle:
        optima = {row['target']: row for row in json.load(handle)}

    paper_circuits()
    paper_objective(control_dir, optima)
    paper_design(filters, amps)
    paper_resolution(filters, base)
    paper_figure2(filters)
    paper_spread(filters, base)
    paper_optimum(filters, base, optima)
    paper_reassignment(optima)
    paper_best_of_n(filters, base, optima)
    paper_amplifier(amps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dir', default=CAMPAIGN_DIR,
                        help='directory holding filter_all.csv, amp_all.csv, baseline.csv')
    parser.add_argument('--control-dir', default=CONTROL_DIR,
                        help='directory holding the matched-budget control CSVs')
    parser.add_argument('--paper', action='store_true',
                        help='print the numbers the ACDSA 2027 paper quotes, in its order '
                             '(re-simulates the few that describe circuits; needs ngspice)')
    args = parser.parse_args()

    if args.paper:
        paper(args.dir, args.control_dir)
        return

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
