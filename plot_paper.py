"""Figures for the paper, drawn from the campaign CSVs.

Kept apart from `analyze_campaign.py` on purpose: that module owns the numbers and
prints them, this one owns the figures. Both read the same CSVs, so a figure cannot
disagree with the table it illustrates.

    python3 plot_paper.py                 # all figures into simulations/figures/
    python3 plot_paper.py --only lottery  # one of them

`--only` takes any of: lottery, bestofn, bode, amplifier, asymmetry, budget.

Nothing here re-runs the GA. `bode` and `asymmetry` do call ngspice, because they
draw responses rather than summarise recorded ones; the other four are pure CSV.
"""
import argparse
import collections
import csv
import json
import os

import numpy as np

import analyze_campaign as ac

CAMPAIGN_DIR = 'simulations/campaign'
CONTROL_DIR = 'simulations/control'
GEN800_DIR = 'simulations/gen800'
OPTIMA = 'simulations/optima_4r.json'
OUT_DIR = 'simulations/figures'

FAMILIES = ('butterworth', 'chebyshev')
CUTOFFS = (1000, 1500, 2000, 3000)
TARGETS = [f'{f}_{c}' for f in FAMILIES for c in CUTOFFS]
GAINS = (5, 10, 15, 20)

# One palette for every figure, so the same thing is the same colour throughout.
C_ANALYTICAL = '#d62728'
C_GA4 = '#1f77b4'
C_GA8 = '#2ca02c'
C_OPTIMUM = '#7f7f7f'
C_TARGET = '#555555'


def _label(target: str) -> str:
    family, _, cutoff = target.rpartition('_')
    return f'{family[:4].capitalize()}\n{int(cutoff) / 1000:g}k'


def _load(path: str) -> list:
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))


def _rmse_by(path: str, variant: str = None) -> dict:
    out = collections.defaultdict(list)
    for row in _load(path):
        if variant is None or row['variant'] == variant:
            out[row['target']].append(float(row['rmse_db']))
    return out


def _baselines() -> dict:
    out = {}
    for row in _load(os.path.join(CAMPAIGN_DIR, 'baseline.csv')):
        variant = '8R' if '8r' in row['ckt_name'] else '4R'
        out[(row['response'], variant)] = float(row['rmse_db'])
    return out


def _optima() -> dict:
    with open(OPTIMA) as handle:
        return {row['target']: row for row in json.load(handle)}


def _save(fig, name: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    fig.savefig(path.replace('.png', '.pdf'), bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


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


def figure_lottery(out_dir: str = OUT_DIR) -> str:
    """The analytical design's quality swings between cells; the GA's does not.

    Three things on one pair of axes, because the argument is the relation between
    them: the GA's 100-seed distribution, the single analytical design, and the true
    global optimum as a floor. The 4R panel is the one where the comparison is fair
    (the optimum is only enumerable there); the 8R panel shows what the finer part
    buys, with no optimum to draw because 10^16 combinations are not enumerable.
    """
    import matplotlib.pyplot as plt

    control = _rmse_by(os.path.join(CONTROL_DIR, 'filter_control.csv'))
    campaign8 = _rmse_by(os.path.join(CAMPAIGN_DIR, 'filter_all.csv'), '8R')
    base = _baselines()
    optima = _optima()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True,
                             constrained_layout=True)
    positions = np.arange(len(TARGETS))

    for ax, data, variant, colour, title in (
            (axes[0], control, '4R', C_GA4, '4R — one X9C103 per resistor'),
            (axes[1], campaign8, '8R', C_GA8, '8R — X9C103 + X9C102 in series')):
        box = ax.boxplot([data[t] for t in TARGETS], positions=positions,
                         widths=0.55, patch_artist=True, showfliers=False,
                         medianprops=dict(color='white', linewidth=1.4),
                         whiskerprops=dict(color=colour), capprops=dict(color=colour))
        for patch in box['boxes']:
            patch.set_facecolor(colour)
            patch.set_alpha(0.75)
            patch.set_edgecolor(colour)

        ax.scatter(positions, [base[(t, variant)] for t in TARGETS],
                   marker='D', s=46, color=C_ANALYTICAL, zorder=5,
                   label='analytical design', edgecolor='white', linewidth=0.7)
        if variant == '4R':
            ax.scatter(positions, [optima[t]['optimum_rmse_db'] for t in TARGETS],
                       marker='_', s=300, color='#111111', linewidth=2.6, zorder=6,
                       label='global optimum (enumerated)')
        ax.set_yscale('log')
        ax.set_xticks(positions)
        ax.set_xticklabels([_label(t) for t in TARGETS], fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.grid(True, axis='y', which='both', linestyle=':', alpha=0.5)

    axes[0].set_ylabel('weighted RMSE [dB]', fontsize=11)
    # One legend below both panels: an in-axes legend sits exactly where the worst
    # analytical points are, and hiding the outlier hides the argument.
    handles, labels = axes[0].get_legend_handles_labels()
    box4, box8 = axes[0].patches[0], axes[1].patches[0]
    fig.legend([box4, box8] + handles, ['GA, 4R (100 seeds)', 'GA, 8R (100 seeds)'] + labels,
               loc='outside lower center', ncol=4, fontsize=9.5, frameon=False)
    fig.suptitle('The analytical design swings between specifications; the GA does not',
                 fontsize=12.5, fontweight='bold')
    return _save(fig, 'fig_lottery.png', out_dir)


def figure_bestofn(out_dir: str = OUT_DIR) -> str:
    """How many independent GA runs it takes to beat the analytical design.

    One run is a coin flip against it; the question a practitioner asks is how many
    restarts buy a result they can rely on. Resampled without replacement from the
    100 observed runs, so the curve is the measured distribution, not a model of it.
    """
    import matplotlib.pyplot as plt

    control = _rmse_by(os.path.join(CONTROL_DIR, 'filter_control.csv'))
    base = _baselines()
    optima = _optima()
    counts = list(range(1, 21))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), constrained_layout=True)

    ax = axes[0]
    # Blues for Butterworth, greens for Chebyshev: red is reserved for the analytical
    # design everywhere in this paper, so no cell may borrow it.
    blues = plt.cm.Blues(np.linspace(0.45, 0.95, len(CUTOFFS)))
    greens = plt.cm.Greens(np.linspace(0.45, 0.95, len(CUTOFFS)))
    for index, target in enumerate(TARGETS):
        family = index // len(CUTOFFS)
        colour = (blues if family == 0 else greens)[index % len(CUTOFFS)]
        ratios = [median_best_of_n(control[target], n) / base[(target, '4R')]
                  for n in counts]
        ax.plot(counts, ratios, '-' if family == 0 else '--', linewidth=1.6,
                color=colour, label=_label(target).replace('\n', ' '))
    ax.axhline(1.0, color=C_ANALYTICAL, linewidth=2.2, zorder=1)
    ax.text(10.5, 1.06, 'the analytical design', color=C_ANALYTICAL, fontsize=9.5,
            ha='center', va='bottom', fontweight='bold')
    ax.set_xlabel('independent GA runs, best kept', fontsize=11)
    ax.set_ylabel('median RMSE / analytical design', fontsize=11)
    ax.set_yscale('log')
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_ylim(top=2.6)
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    ax.legend(fontsize=7.5, ncol=2, loc='lower left', framealpha=0.95)
    ax.set_title('below the red line, the GA wins', fontsize=11)

    ax = axes[1]
    wins, ratio_mid = [], []
    for n in counts:
        won, ratios = 0, []
        for target in TARGETS:
            median = median_best_of_n(control[target], n)
            won += median < base[(target, '4R')]
            ratios.append(median / optima[target]['optimum_rmse_db'])
        wins.append(won)
        ratio_mid.append(np.median(ratios))
    ax.bar(counts, wins, color=C_GA4, alpha=0.30, zorder=1)
    ax.set_xlabel('independent GA runs, best kept', fontsize=11)
    ax.set_ylabel('specifications won (of 8)', fontsize=11, color=C_GA4)
    ax.set_yticks(range(0, 9, 2))
    ax.set_ylim(0, 8.6)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.tick_params(axis='y', colors=C_GA4)
    ax.grid(True, axis='y', linestyle=':', alpha=0.5)
    ax.annotate('every specification\nwon from here', xy=(10, 8), xytext=(13.2, 5.0),
                fontsize=9.5, color=C_GA4, fontweight='bold', ha='center',
                arrowprops=dict(arrowstyle='->', color=C_GA4, linewidth=1.4,
                                connectionstyle='arc3,rad=0.15'))

    twin = ax.twinx()
    # Median across cells only. The spread across cells is the left panel's job; drawn
    # here as a band it just sits on top of the bars and reads as noise.
    twin.plot(counts, ratio_mid, color='#8c564b', linewidth=2.4, zorder=4)
    twin.set_ylabel('median × the global optimum', fontsize=10.5, color='#8c564b')
    twin.tick_params(axis='y', colors='#8c564b')
    twin.set_ylim(bottom=1.0)
    ax.set_title('ten runs win every specification', fontsize=11)
    return _save(fig, 'fig_bestofn.png', out_dir)


def figure_bode(out_dir: str = OUT_DIR, target: str = 'chebyshev_3000') -> str:
    """The response itself, for the cell where the analytical design lands worst.

    Fitness numbers are only believable next to a curve. Both realizations of the
    same specification are drawn against the ideal response, with the analytical
    design for contrast, and the lower panel carries the error that the objective
    actually integrates.
    """
    import matplotlib.pyplot as plt

    import filter_baseline
    import filter_sk4_evolution as sk4
    import filter_sk8_evolution as sk8

    def response(module, taps):
        spec = module.SPEC
        circuit = spec.circuit_factory()
        spec.setup_hook(circuit, target)
        return spec.evaluator.metrics(circuit, spec.resistor_mapper(taps), target)

    def best_taps(path, variant):
        rows = [r for r in _load(path)
                if r['target'] == target and r['variant'] == variant]
        best = min(rows, key=lambda r: float(r['rmse_db']))
        return json.loads(best['solution'])

    ga4 = response(sk4, best_taps(os.path.join(CONTROL_DIR, 'filter_control.csv'), '4R'))
    ga8 = response(sk8, best_taps(os.path.join(CAMPAIGN_DIR, 'filter_all.csv'), '8R'))
    analytical = response(sk4, filter_baseline.design(sk4, target)['taps'])

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1.4]},
                             constrained_layout=True)
    freqs = ga8['freqs']
    ideal = ga8['target_db']

    ax = axes[0]
    ax.semilogx(freqs, ideal, color=C_TARGET, linestyle='--', linewidth=2.0,
                label='ideal response', zorder=2)
    ax.semilogx(freqs, analytical['response_db'], color=C_ANALYTICAL, linewidth=1.5,
                label=f"analytical, 4R ({analytical['rmse_db']:.4f} dB)", zorder=3)
    ax.semilogx(freqs, ga4['response_db'], color=C_GA4, linewidth=1.5,
                label=f"GA, 4R ({ga4['rmse_db']:.4f} dB)", zorder=4)
    ax.semilogx(freqs, ga8['response_db'], color=C_GA8, linewidth=1.5,
                label=f"GA, 8R ({ga8['rmse_db']:.4f} dB)", zorder=5)
    ax.set_ylabel('|H(f)| [dB]', fontsize=11)
    ax.set_ylim(-90, 5)
    ax.legend(fontsize=8.5, loc='lower left')
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    ax.set_title(f'{target.replace("_", " ")} Hz, 4th order', fontsize=11,
                 fontweight='bold')

    # At full scale the four curves are one line, which is itself the point: these are
    # all good filters and the comparison lives two orders of magnitude below what a
    # Bode plot resolves. The inset is where the reader can actually see them differ.
    cutoff = ga8['cutoff_target_hz']
    inset = ax.inset_axes([0.545, 0.46, 0.43, 0.36])
    for curve, colour, width in ((ideal, C_TARGET, 2.0),
                                 (analytical['response_db'], C_ANALYTICAL, 1.4),
                                 (ga4['response_db'], C_GA4, 1.4),
                                 (ga8['response_db'], C_GA8, 1.4)):
        inset.semilogx(freqs, curve, color=colour, linewidth=width,
                       linestyle='--' if colour == C_TARGET else '-')
    inset.set_xlim(0.2 * cutoff, 1.6 * cutoff)
    inset.set_ylim(-3.2, 1.0)
    inset.grid(True, which='both', linestyle=':', alpha=0.5)
    # A log axis auto-labels every decade multiple here and they overlap; name the
    # few that matter instead, in kHz.
    ticks = [t for t in (0.25, 0.5, 1.0, 1.5) if 0.2 <= t <= 1.6]
    inset.set_xticks([t * cutoff for t in ticks])
    inset.set_xticklabels([f'{t * cutoff / 1000:g}k' for t in ticks])
    inset.set_xticks([], minor=True)
    inset.tick_params(labelsize=7)
    inset.text(0.03, 0.06, 'passband and corner', transform=inset.transAxes,
               fontsize=8, va='bottom')

    ax = axes[1]
    for metrics, colour in ((analytical, C_ANALYTICAL), (ga4, C_GA4), (ga8, C_GA8)):
        ax.semilogx(freqs, metrics['response_db'] - ideal, color=colour, linewidth=1.4)
    ax.axhline(0, color=C_TARGET, linewidth=0.9)
    ax.set_xlabel('frequency [Hz]', fontsize=11)
    ax.set_ylabel('error [dB]', fontsize=11)
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    # The passband is where the objective weights error most heavily.
    ax.axvspan(freqs[0], ga8['cutoff_target_hz'], color=C_GA4, alpha=0.07)
    return _save(fig, 'fig_bode.png', out_dir)


def figure_amplifier(out_dir: str = OUT_DIR) -> str:
    """The amplifier fails by a different mechanism: a floor plus a trap.

    Its error distribution is bimodal, not spread, so a box plot would describe it
    badly -- every run lands either on a floor the resistors cannot move or in a
    local optimum near 9%. Drawing the 100 seeds individually shows that, and shows
    that the finer part empties the trap without lowering the floor.
    """
    import matplotlib.pyplot as plt

    campaign = collections.defaultdict(list)
    for row in _load(os.path.join(CAMPAIGN_DIR, 'amp_all.csv')):
        campaign[(int(float(row['desired_gain'])), row['variant'])].append(
            float(row['avg_error_percent']))
    control = collections.defaultdict(list)
    for row in _load(os.path.join(CONTROL_DIR, 'amp_control.csv')):
        control[int(float(row['desired_gain']))].append(float(row['avg_error_percent']))

    fig, ax = plt.subplots(figsize=(8.4, 4.6), constrained_layout=True)
    rng = np.random.default_rng(1)
    arms = (('4R', C_GA4, -0.26), ('4R @ pop 40', C_GA4, 0.0), ('8R', C_GA8, 0.26))

    for index, gain in enumerate(GAINS):
        series = (campaign[(gain, '4R')], control[gain], campaign[(gain, '8R')])
        for (name, colour, offset), values in zip(arms, series):
            values = np.array(values)
            jitter = rng.uniform(-0.075, 0.075, values.size)
            ax.scatter(index + offset + jitter, values, s=7, alpha=0.45,
                       color=colour, edgecolors='none',
                       marker='o' if 'pop' not in name else '^',
                       label=name if index == 0 else None)
            ax.plot([index + offset - 0.1, index + offset + 0.1],
                    [np.median(values)] * 2, color='black', linewidth=1.6, zorder=5)

    # The analytical design, which every one of the 800 runs beats.
    import campaign as campaign_module
    analytical = []
    for gain in GAINS:
        spec = campaign_module._module('amp', '4R').SPEC
        circuit = spec.circuit_factory()
        spec.setup_hook(circuit, gain)
        taps = campaign_module.baseline_taps('amp', '4R', gain)
        analytical.append(spec.evaluator.metrics(
            circuit, spec.resistor_mapper(taps), gain)['avg_error_percent'])
    ax.scatter(range(len(GAINS)), analytical, marker='D', s=52, color=C_ANALYTICAL,
               zorder=6, edgecolor='white', linewidth=0.7, label='analytical design')

    ax.set_xticks(range(len(GAINS)))
    ax.set_xticklabels([f'gain {g}' for g in GAINS], fontsize=10)
    ax.set_ylabel('mean gain error [%]', fontsize=11)
    ax.grid(True, axis='y', linestyle=':', alpha=0.5)
    ax.set_ylim(0.5, 15.5)
    # Name the two regimes: the whole point is that they are regimes, not a spread.
    ax.annotate('local optimum:\nwrong bias point', xy=(0.74, 9.0), xytext=(0.36, 6.2),
                fontsize=8.5, color='#444444', ha='center',
                arrowprops=dict(arrowstyle='->', color='#888888', linewidth=1.1))
    ax.annotate('floor set by the small-signal model,\nwhich no resistor resolution moves',
                xy=(1.74, 3.45), xytext=(2.15, 5.9), fontsize=8.5, color='#444444',
                ha='center',
                arrowprops=dict(arrowstyle='->', color='#888888', linewidth=1.1))
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=4, fontsize=9.5,
               markerscale=1.8, frameon=False)
    ax.set_title('Every run lands on a floor or in a trap — and every run beats the '
                 'analytical design', fontsize=11, fontweight='bold')
    return _save(fig, 'fig_amplifier.png', out_dir)


def figure_asymmetry(out_dir: str = OUT_DIR) -> str:
    """Swapping R1 and R2 changes the response, which the design equations cannot say.

    omega_o and Q depend on R1 and R2 only through their product and sum, so the
    analytical route picks an order arbitrarily. With a real op-amp the two orders
    are not equivalent, and the difference is worth up to 1.84x -- free, since it is
    the same components on the same board.
    """
    import matplotlib.pyplot as plt

    import filter_baseline
    import filter_sk4_evolution as sk4
    optima = _optima()

    def rmse(taps, target):
        circuit = sk4.SPEC.circuit_factory()
        sk4.SPEC.setup_hook(circuit, target)
        return 1.0 / sk4.SPEC.evaluator.fitness(
            circuit, sk4.SPEC.resistor_mapper(list(taps)), target) - 1.0

    as_designed, best_order, optimum = [], [], []
    for target in TARGETS:
        taps = filter_baseline.design(sk4, target)['taps']
        orders = [(taps[0], taps[1], taps[2], taps[3]),
                  (taps[1], taps[0], taps[2], taps[3]),
                  (taps[0], taps[1], taps[3], taps[2]),
                  (taps[1], taps[0], taps[3], taps[2])]
        scores = [rmse(o, target) for o in orders]
        as_designed.append(scores[0])
        best_order.append(min(scores))
        optimum.append(optima[target]['optimum_rmse_db'])

    fig, ax = plt.subplots(figsize=(8.8, 4.6), constrained_layout=True)
    positions = np.arange(len(TARGETS))
    # Two stacked segments, not two overlaid bars: the upper one is what swapping the
    # two resistors recovers for free, the lower one is what is left for a search.
    free = '#ff7f0e'
    for x, a, b, c in zip(positions, as_designed, best_order, optimum):
        ax.plot([x, x], [b, a], color=free, linewidth=7, solid_capstyle='butt', zorder=2)
        ax.plot([x, x], [c, b], color='#b0b0b0', linewidth=7, solid_capstyle='butt',
                zorder=1)
    ax.scatter(positions, as_designed, marker='D', s=54, color=C_ANALYTICAL, zorder=5,
               edgecolor='white', linewidth=0.8, label='analytical design, as written')
    ax.scatter(positions, best_order, marker='o', s=46, color='#8c4a00', zorder=5,
               edgecolor='white', linewidth=0.8,
               label='same components, best R1/R2 ordering')
    ax.scatter(positions, optimum, marker='_', s=300, color='#111111', linewidth=2.6,
               zorder=6, label='global optimum')
    ax.plot([], [], color=free, linewidth=7, label='recovered by reordering alone')
    ax.plot([], [], color='#b0b0b0', linewidth=7, label='still requires search')
    ax.set_yscale('log')
    ax.set_xticks(positions)
    ax.set_xticklabels([_label(t) for t in TARGETS], fontsize=8)
    ax.set_ylabel('weighted RMSE [dB]', fontsize=11)
    ax.grid(True, axis='y', which='both', linestyle=':', alpha=0.5)
    # Cells where the swap alone lands on the optimum: one label, an arrow to each.
    reached = [x for x, b, c in zip(positions, best_order, optimum) if b / c < 1.001]
    if reached:
        anchor = (float(np.mean(reached)), min(optimum[x] for x in reached) * 0.42)
        for x in reached:
            ax.annotate('', xy=(x, optimum[x] * 0.93), xytext=anchor,
                        arrowprops=dict(arrowstyle='->', color='#8c4a00', linewidth=1.1))
        ax.text(anchor[0], anchor[1] * 0.93, 'reordering alone reaches the optimum',
                fontsize=8.5, ha='center', va='top', color='#8c4a00', fontweight='bold')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3, fontsize=9,
               frameon=False)
    ax.set_title('The design equations are symmetric in R1 and R2; the real circuit is not',
                 fontsize=11.5, fontweight='bold')
    return _save(fig, 'fig_asymmetry.png', out_dir)


def figure_budget(out_dir: str = OUT_DIR) -> str:
    """Separating what the search budget bought from what the resistor bought.

    The two variants ship with different populations, so the plain 4R-vs-8R gap mixes
    the two. The middle arm is 4R re-run at 8R's population: the step to it is
    budget, the step past it is resolution.
    """
    import matplotlib.pyplot as plt

    base4 = _rmse_by(os.path.join(CAMPAIGN_DIR, 'filter_all.csv'), '4R')
    matched = _rmse_by(os.path.join(CONTROL_DIR, 'filter_control.csv'))
    better = _rmse_by(os.path.join(CAMPAIGN_DIR, 'filter_all.csv'), '8R')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

    ax = axes[0]
    positions = np.arange(len(TARGETS))
    arms = ((base4, '4R, pop 20', C_GA4, 'D', -0.22),
            (matched, '4R, pop 40', C_GA4, '^', 0.0),
            (better, '8R, pop 40', C_GA8, 'o', 0.22))
    for data, name, colour, marker, offset in arms:
        medians = [np.median(data[t]) for t in TARGETS]
        ax.scatter(positions + offset, medians, marker=marker, s=44, color=colour,
                   alpha=0.55 if marker == '^' else 1.0, label=name,
                   edgecolor='white', linewidth=0.6, zorder=4)
    for x, t in zip(positions, TARGETS):
        ax.plot([x - 0.22, x, x + 0.22],
                [np.median(base4[t]), np.median(matched[t]), np.median(better[t])],
                color='#bbbbbb', linewidth=1.2, zorder=2)
    ax.set_xticks(positions)
    ax.set_xticklabels([_label(t) for t in TARGETS], fontsize=8)
    ax.set_ylabel('median weighted RMSE [dB]', fontsize=11)
    ax.set_yscale('log')
    ax.grid(True, axis='y', which='both', linestyle=':', alpha=0.5)
    ax.legend(fontsize=9)
    ax.set_title('per specification', fontsize=11)

    ax = axes[1]
    pooled = [np.median(np.concatenate([data[t] for t in TARGETS]))
              for data, *_ in arms]
    names = [name for _, name, *_ in arms]
    bars = ax.bar(names, pooled, color=[C_GA4, C_GA4, C_GA8], width=0.55)
    bars[1].set_alpha(0.55)
    for bar, value in zip(bars, pooled):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f'{value:.4f}',
                ha='center', va='bottom', fontsize=9)
    ax.annotate(f'budget\n{pooled[0] / pooled[1]:.2f}x', xy=(0.5, max(pooled) * 0.75),
                ha='center', fontsize=10, color='#444444')
    ax.annotate(f'resolution\n{pooled[1] / pooled[2]:.2f}x', xy=(1.5, max(pooled) * 0.75),
                ha='center', fontsize=10, color='#444444', fontweight='bold')
    ax.set_ylabel('median weighted RMSE [dB]', fontsize=11)
    ax.grid(True, axis='y', linestyle=':', alpha=0.5)
    ax.set_title('pooled over 800 runs each', fontsize=11)
    return _save(fig, 'fig_budget.png', out_dir)


FIGURES = {
    'lottery': figure_lottery,
    'bestofn': figure_bestofn,
    'bode': figure_bode,
    'amplifier': figure_amplifier,
    'asymmetry': figure_asymmetry,
    'budget': figure_budget,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--only', action='append', choices=sorted(FIGURES),
                        help='Draw only this figure; repeat for several')
    parser.add_argument('--out-dir', default=OUT_DIR)
    args = parser.parse_args()

    for name in (args.only or sorted(FIGURES)):
        path = FIGURES[name](out_dir=args.out_dir)
        print(f'  {name:<10} -> {path}', flush=True)


if __name__ == '__main__':
    main()
