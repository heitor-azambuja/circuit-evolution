import data_parse
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os


AVG_ERROR_SUCCESS_THRESHOLD = 5.0
MAX_ERROR_SUCCESS_THRESHOLD = 10.0


def parse_float(value, default=np.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def summarize_group(rows):
    avg_errors = np.array([parse_float(row.get('avg_error_percent')) for row in rows], dtype=float)
    max_errors = np.array([parse_float(row.get('max_error_percent')) for row in rows], dtype=float)

    avg_errors = avg_errors[~np.isnan(avg_errors)]
    max_errors = max_errors[~np.isnan(max_errors)]

    if avg_errors.size == 0:
        return None

    max_errors_for_success = max_errors if max_errors.size == avg_errors.size else np.full(avg_errors.size, np.inf)
    success_mask = (
        (avg_errors <= AVG_ERROR_SUCCESS_THRESHOLD) &
        (max_errors_for_success <= MAX_ERROR_SUCCESS_THRESHOLD)
    )

    best_idx = int(np.argmin(avg_errors))
    best_row = rows[best_idx]
    best_seed = best_row.get('seed', '')

    summary = {
        'count': int(avg_errors.size),
        'best_avg_error_percent': float(np.min(avg_errors)),
        'best_exec': best_row.get('exec_counter'),
        'best_seed': best_seed,
        'mean_avg_error_percent': float(np.mean(avg_errors)),
        'median_avg_error_percent': float(np.median(avg_errors)),
        'iqr_avg_error_percent': float(np.percentile(avg_errors, 75) - np.percentile(avg_errors, 25)),
        'p90_avg_error_percent': float(np.percentile(avg_errors, 90)),
        'p95_avg_error_percent': float(np.percentile(avg_errors, 95)),
        'success_rate_percent': float(np.mean(success_mask) * 100.0),
    }

    if max_errors.size > 0:
        summary.update({
            'mean_max_error_percent': float(np.mean(max_errors)),
            'median_max_error_percent': float(np.median(max_errors)),
            'p90_max_error_percent': float(np.percentile(max_errors, 90)),
            'p95_max_error_percent': float(np.percentile(max_errors, 95)),
        })
    else:
        summary.update({
            'mean_max_error_percent': np.nan,
            'median_max_error_percent': np.nan,
            'p90_max_error_percent': np.nan,
            'p95_max_error_percent': np.nan,
        })

    summary['quality_score'] = (
        0.6 * summary['median_avg_error_percent'] +
        0.3 * summary['p95_max_error_percent'] +
        0.1 * (100.0 - summary['success_rate_percent'])
    )
    return summary


def summarize_convergence(rows):
    best_gens = np.array([parse_float(row.get('best_generation')) for row in rows], dtype=float)
    best_gens = best_gens[~np.isnan(best_gens)]

    if best_gens.size == 0:
        return None

    return {
        'count': int(best_gens.size),
        'mean_best_generation': float(np.mean(best_gens)),
        'median_best_generation': float(np.median(best_gens)),
        'p90_best_generation': float(np.percentile(best_gens, 90)),
    }


def summarize_runtime(rows):
    runtime = np.array([parse_float(row.get('runtime_s')) for row in rows], dtype=float)
    runtime = runtime[~np.isnan(runtime)]

    if runtime.size == 0:
        return None

    return {
        'count': int(runtime.size),
        'mean_runtime_s': float(np.mean(runtime)),
        'median_runtime_s': float(np.median(runtime)),
        'p95_runtime_s': float(np.percentile(runtime, 95)),
    }


def group_rows(simulation_data):
    grouped = {}
    for row in simulation_data:
        ckt_name = row.get('ckt_name')
        desired_gain = parse_int(row.get('desired_gain'))
        if ckt_name is None or desired_gain is None:
            continue
        key = (ckt_name, desired_gain)
        grouped.setdefault(key, []).append(row)
    return grouped


def plot_group_summary(ckt_name: str, desired_gain: int, rows: list, out_dir: str = 'simulations') -> None:
    """
    Save a 3-panel summary figure for one (circuit, gain) group:
      Top-left : box + strip plot of avg_error_percent per run
      Top-right: box + strip plot of max_error_percent per run
      Bottom   : best_generation scatter (convergence speed) per run
    """
    avg_errors = np.array([parse_float(r.get('avg_error_percent')) for r in rows], dtype=float)
    max_errors = np.array([parse_float(r.get('max_error_percent')) for r in rows], dtype=float)
    best_gens  = np.array([parse_float(r.get('best_generation'))   for r in rows], dtype=float)
    execs      = np.array([parse_float(r.get('exec_counter'))      for r in rows], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
    title = f'{ckt_name}  |  gain={desired_gain}  |  n={len(rows)}'
    fig.suptitle(title, fontsize=13, fontweight='bold')

    BLUE  = '#1f77b4'
    GREEN = '#2ca02c'
    RED   = '#d62728'

    # --- Panel 1: avg_error_percent ---
    ax = axes[0]
    valid = avg_errors[~np.isnan(avg_errors)]
    if valid.size:
        bp = ax.boxplot(valid, widths=0.4, patch_artist=True, showfliers=False,
                        medianprops=dict(color='white', linewidth=2))
        bp['boxes'][0].set_facecolor(BLUE)
        bp['boxes'][0].set_alpha(0.6)
        jitter = np.random.default_rng(0).uniform(-0.12, 0.12, size=valid.size)
        ax.scatter(np.ones(valid.size) + jitter, valid, color=BLUE, s=30, alpha=0.7, zorder=5)
        ax.axhline(AVG_ERROR_SUCCESS_THRESHOLD, color=RED, linestyle='--',
                   linewidth=1.2, label=f'threshold {AVG_ERROR_SUCCESS_THRESHOLD}%')
        ax.legend(fontsize=8)
    ax.set_xticks([])
    ax.set_ylabel('avg error [%]', fontsize=11)
    ax.set_title('Avg Error Distribution', fontsize=11)
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    # --- Panel 2: max_error_percent ---
    ax = axes[1]
    valid_max = max_errors[~np.isnan(max_errors)]
    if valid_max.size:
        bp2 = ax.boxplot(valid_max, widths=0.4, patch_artist=True, showfliers=False,
                         medianprops=dict(color='white', linewidth=2))
        bp2['boxes'][0].set_facecolor(GREEN)
        bp2['boxes'][0].set_alpha(0.6)
        jitter2 = np.random.default_rng(1).uniform(-0.12, 0.12, size=valid_max.size)
        ax.scatter(np.ones(valid_max.size) + jitter2, valid_max, color=GREEN, s=30, alpha=0.7, zorder=5)
        ax.axhline(MAX_ERROR_SUCCESS_THRESHOLD, color=RED, linestyle='--',
                   linewidth=1.2, label=f'threshold {MAX_ERROR_SUCCESS_THRESHOLD}%')
        ax.legend(fontsize=8)
    ax.set_xticks([])
    ax.set_ylabel('max error [%]', fontsize=11)
    ax.set_title('Max Error Distribution', fontsize=11)
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    # --- Panel 3: convergence generation per run ---
    ax = axes[2]
    mask = ~np.isnan(best_gens)
    if mask.any():
        x = execs[mask] if not np.isnan(execs[mask]).all() else np.arange(1, mask.sum() + 1)
        ax.scatter(x, best_gens[mask], color='#ff7f0e', s=40, alpha=0.8, zorder=5, label='best gen')
        ax.axhline(np.median(best_gens[mask]), color='#ff7f0e', linestyle='--',
                   linewidth=1.2, alpha=0.7, label=f'median={np.median(best_gens[mask]):.0f}')
        ax.legend(fontsize=8)
    ax.set_xlabel('execution #', fontsize=11)
    ax.set_ylabel('best generation', fontsize=11)
    ax.set_title('Convergence Speed per Run', fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.6)

    safe_name = ckt_name.replace(' ', '_')
    path = os.path.join(out_dir, f'summary_{safe_name}_gain{desired_gain}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [plot] {path}')


def plot_comparison_overview(grouped: dict, group_summaries: list, out_dir: str = 'simulations') -> None:
    """
    Save a single overview figure comparing all (circuit, gain) groups:
      - Grouped bar chart: median avg_error_percent per group, coloured by circuit type
      - Error bars show IQR
      - Overlaid success-rate as a line on a secondary axis
    """
    if not group_summaries:
        return

    labels   = [f"{s['ckt_name'].replace('bjt_class_a_amp_', '')} g={s['desired_gain']}"
                for s in group_summaries]
    medians  = [s['median_avg_error_percent']   for s in group_summaries]
    iqr_low  = [s['median_avg_error_percent'] - s['iqr_avg_error_percent'] / 2 for s in group_summaries]
    iqr_high = [s['median_avg_error_percent'] + s['iqr_avg_error_percent'] / 2 for s in group_summaries]
    successes = [s['success_rate_percent']       for s in group_summaries]
    colors   = ['#1f77b4' if '4r' in s['ckt_name'] else '#ff7f0e' for s in group_summaries]

    x = np.arange(len(labels))
    err_low  = np.maximum(0, np.array(medians) - np.array(iqr_low))
    err_high = np.array(iqr_high) - np.array(medians)

    fig, ax1 = plt.subplots(figsize=(max(8, len(labels) * 1.4), 5), constrained_layout=True)

    bars = ax1.bar(x, medians, color=colors, alpha=0.75, width=0.55, zorder=3,
                   yerr=[err_low, err_high], capsize=4, error_kw=dict(ecolor='#444', lw=1.2))
    ax1.axhline(AVG_ERROR_SUCCESS_THRESHOLD, color='#d62728', linestyle='--',
                linewidth=1.2, label=f'avg_err threshold ({AVG_ERROR_SUCCESS_THRESHOLD}%)')
    ax1.set_ylabel('median avg error [%]', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax1.grid(True, axis='y', linestyle=':', alpha=0.6, zorder=0)
    ax1.set_title('Comparison Overview — median avg error + success rate', fontsize=12, fontweight='bold')

    ax2 = ax1.twinx()
    ax2.plot(x, successes, color='#2ca02c', marker='o', linewidth=1.8,
             markersize=6, label='success rate %', zorder=6)
    ax2.set_ylabel('success rate [%]', fontsize=11, color='#2ca02c')
    ax2.tick_params(axis='y', labelcolor='#2ca02c')
    ax2.set_ylim(0, 110)

    # Legend combining both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    # colour legend for circuit types
    from matplotlib.patches import Patch
    handles1 += [Patch(facecolor='#1f77b4', alpha=0.75, label='4R circuit'),
                 Patch(facecolor='#ff7f0e', alpha=0.75, label='8R circuit')]
    ax1.legend(handles=handles1 + handles2, labels=labels1 + ['4R circuit', '8R circuit'] + labels2,
               fontsize=9, loc='upper left')

    path = os.path.join(out_dir, 'overview_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [plot] {path}')


def print_quality_report(group_summaries):
    print('\n=== Quality Ranking (lower score is better) ===')
    print(f'Success threshold: avg_error_percent <= {AVG_ERROR_SUCCESS_THRESHOLD:.2f}%, '
          f'max_error_percent <= {MAX_ERROR_SUCCESS_THRESHOLD:.2f}%')

    for summary in sorted(group_summaries, key=lambda item: item['quality_score']):
        print(
            f"\n{summary['ckt_name']} gain={summary['desired_gain']} | runs={summary['count']} | "
            f"score={summary['quality_score']:.2f}"
        )
        print(
            f"  best={summary['best_avg_error_percent']:.2f}% (exec={summary['best_exec']}, "
            f"seed={summary['best_seed']}), "
            f"mean={summary['mean_avg_error_percent']:.2f}%, median={summary['median_avg_error_percent']:.2f}%"
        )
        print(
            f"  iqr={summary['iqr_avg_error_percent']:.2f}%, p90={summary['p90_avg_error_percent']:.2f}%, "
            f"p95={summary['p95_avg_error_percent']:.2f}%"
        )
        print(
            f"  max_error mean={summary['mean_max_error_percent']:.2f}%, "
            f"p90={summary['p90_max_error_percent']:.2f}%, p95={summary['p95_max_error_percent']:.2f}%"
        )
        print(f"  success_rate={summary['success_rate_percent']:.1f}%")
        conv = summary.get('convergence')
        if conv:
            print(
                f"  convergence: median_gen={conv['median_best_generation']:.0f}, "
                f"p90_gen={conv['p90_best_generation']:.0f} (n={conv['count']})"
            )


def print_significance_report(grouped):
    print('\n=== Statistical Significance: 4R vs 8R per gain (Mann-Whitney U) ===')
    gains = sorted({gain for (_, gain) in grouped.keys()})
    found_any = False
    for gain in gains:
        rows_4r = grouped.get(('bjt_class_a_amp_4r', gain), [])
        rows_8r = grouped.get(('bjt_class_a_amp_8r', gain), [])
        errs_4r = np.array([parse_float(r.get('avg_error_percent')) for r in rows_4r], dtype=float)
        errs_8r = np.array([parse_float(r.get('avg_error_percent')) for r in rows_8r], dtype=float)
        errs_4r = errs_4r[~np.isnan(errs_4r)]
        errs_8r = errs_8r[~np.isnan(errs_8r)]
        if errs_4r.size < 2 or errs_8r.size < 2:
            continue
        found_any = True
        stat, p = stats.mannwhitneyu(errs_4r, errs_8r, alternative='two-sided')
        sig = '✓ significant' if p < 0.05 else '✗ not significant'
        better = '4R' if np.median(errs_4r) < np.median(errs_8r) else '8R'
        print(
            f"  gain={gain} | U={stat:.0f}, p={p:.4f} [{sig}] | "
            f"lower median: {better} "
            f"(4R median={np.median(errs_4r):.2f}%, 8R median={np.median(errs_8r):.2f}%)"
        )
    if not found_any:
        print('  Insufficient data — need n>=2 runs for both 4R and 8R at same gain.')


def print_runtime_report(runtime_summaries):
    print('\n=== Runtime Summary (non-ranking) ===')
    if not runtime_summaries:
        print('No runtime data found (missing `runtime_s` values).')
        return

    for summary in sorted(runtime_summaries, key=lambda item: item['mean_runtime_s']):
        print(
            f"{summary['ckt_name']} gain={summary['desired_gain']} | samples={summary['count']} | "
            f"mean={summary['mean_runtime_s']:.2f}s | median={summary['median_runtime_s']:.2f}s | "
            f"p95={summary['p95_runtime_s']:.2f}s"
        )

if __name__ == "__main__":
    simulation_data = data_parse.load_csv_to_json('simulations/data.csv')

    grouped = group_rows(simulation_data)

    group_summaries = []
    runtime_summaries = []

    for (ckt_name, desired_gain), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        summary = summarize_group(rows)
        if summary is not None:
            summary['ckt_name'] = ckt_name
            summary['desired_gain'] = desired_gain
            conv = summarize_convergence(rows)
            if conv is not None:
                summary['convergence'] = conv
            group_summaries.append(summary)

        runtime_summary = summarize_runtime(rows)
        if runtime_summary is not None:
            runtime_summary['ckt_name'] = ckt_name
            runtime_summary['desired_gain'] = desired_gain
            runtime_summaries.append(runtime_summary)

    if not group_summaries:
        print('No valid simulation rows were found in simulations/data.csv')
    else:
        print_quality_report(group_summaries)
        print_significance_report(grouped)
        print('\nGenerating plots...')
        for (ckt_name, desired_gain), rows in grouped.items():
            plot_group_summary(ckt_name, desired_gain, rows)
        plot_comparison_overview(grouped, group_summaries)

    print_runtime_report(runtime_summaries)