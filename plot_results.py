"""
Regenerate all simulation plots from saved data — no re-running the GA needed.

Usage:
    python plot_results.py                  # regenerate everything
    python plot_results.py --waveforms      # waveform plots only
    python plot_results.py --fitness        # fitness history plots only
    python plot_results.py --summary        # evaluate_sims summary plots only
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import circuits
import components
import data_parse
import evaluate_sims


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_amp(ckt_name: str):
    if '8r' in ckt_name:
        amp = circuits.BJTClassAAmp8R()
    else:
        amp = circuits.BJTClassAAmp()
    amp.configure_capacitors(47, 100, 47)
    return amp


def _get_resistances(row: dict, ckt_name: str):
    """Parse resistors field from a CSV row back to a plain float list."""
    raw = row.get('resistors') or row.get('solution')
    if raw is None:
        return None
    if isinstance(raw, list):
        return [float(v) for v in raw]
    try:
        parsed = json.loads(raw)
        return [float(v) for v in parsed]
    except (json.JSONDecodeError, TypeError):
        return None


_STEP_TIME_FINE = 0.000005


def _compute_waveform(amp, resistances: list, desired_gain: float) -> dict:
    amp.configure_resistors(resistances)
    analysis = amp.transient_analysis(step_time=_STEP_TIME_FINE)
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


# ---------------------------------------------------------------------------
# Waveform plots
# ---------------------------------------------------------------------------

def regenerate_waveforms(rows: list, out_dir: str = 'simulations') -> None:
    print('=== Regenerating waveform plots ===')
    pot_100k = components.DigitalPot(100_000, 100)
    pot_10k  = components.DigitalPot(10_000, 100)

    for row in rows:
        ckt_name    = row.get('ckt_name', '')
        desired_gain = evaluate_sims.parse_float(row.get('desired_gain'))
        exec_counter = row.get('exec_counter', '?')
        seed         = row.get('seed', '')

        resistances = _get_resistances(row, ckt_name)
        if resistances is None or desired_gain is None:
            print(f'  [skip] exec={exec_counter} — missing resistors or gain')
            continue

        try:
            amp = _build_amp(ckt_name)
            metrics = _compute_waveform(amp, resistances, desired_gain)
        except Exception as exc:
            print(f'  [error] exec={exec_counter}: {exc}')
            continue

        is_8r  = '8r' in ckt_name
        label  = '(8R)' if is_8r else ''
        t, out, des = metrics['time_us'], metrics['output'], metrics['desired']

        fig, axes = plt.subplots(2, 1, figsize=(10, 7),
                                 gridspec_kw={'height_ratios': [3, 1]},
                                 constrained_layout=True)

        ax = axes[0]
        ax.plot(t, des, color='#2ca02c', linewidth=1.8, linestyle='--', label='desired', zorder=3)
        ax.plot(t, out, color='#1f77b4', linewidth=1.8, label='output', zorder=4)
        ax.fill_between(t, out, des, alpha=0.18, color='#d62728', label='error')
        ax.set_xlabel('time [µs]', fontsize=11)
        ax.set_ylabel('voltage [mV]', fontsize=11)
        ax.set_title(
            f'BJT Class A Amplifier {label} — Gain = {desired_gain:.0f}   '
            f'(exec {exec_counter}, seed {seed})',
            fontsize=12, fontweight='bold'
        )
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, linestyle=':', alpha=0.6)

        res_str = ', '.join(f'{r/1000:.1f}k' for r in metrics['resistances'])
        annotation = (
            f"avg err: {metrics['avg_error_percent']:.2f}%\n"
            f"max err: {metrics['max_error_percent']:.2f}%\n"
            f"R: [{res_str}] Ω"
        )
        ax.text(0.01, 0.97, annotation, transform=ax.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          alpha=0.8, edgecolor='#aaaaaa'))

        axe = axes[1]
        axe.fill_between(t, np.abs(out - des), color='#d62728', alpha=0.5, linewidth=0)
        axe.plot(t, np.abs(out - des), color='#d62728', linewidth=1.0)
        axe.set_xlabel('time [µs]', fontsize=11)
        axe.set_ylabel('|error| [mV]', fontsize=11)
        axe.grid(True, linestyle=':', alpha=0.6)

        path = os.path.join(out_dir, f'{ckt_name}_gain{desired_gain:.0f}_execution{exec_counter}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  [saved] {path}')


# ---------------------------------------------------------------------------
# Fitness history plots
# ---------------------------------------------------------------------------

def regenerate_fitness_plots(out_dir: str = 'simulations') -> None:
    print('=== Regenerating fitness history plots ===')
    pattern = os.path.join(out_dir, 'fitness_history_*.json')
    files = sorted(glob.glob(pattern))

    if not files:
        print('  No fitness_history_*.json files found — run the GA first.')
        return

    for path in files:
        with open(path) as f:
            data = json.load(f)

        history     = data.get('best_solutions_fitness', [])
        ckt_name    = data.get('ckt_name', 'unknown')
        desired_gain = data.get('desired_gain', '?')
        exec_counter = data.get('exec_counter', '?')

        if not history:
            continue

        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        ax.plot(history, linewidth=2, color='#1f77b4')
        ax.set_xlabel('Generation', fontsize=11)
        ax.set_ylabel('Fitness', fontsize=11)
        ax.set_title(
            f'Fitness — {ckt_name} gain={desired_gain} exec={exec_counter}',
            fontsize=11
        )
        ax.grid(True, linestyle=':', alpha=0.6)

        save_path = os.path.join(
            out_dir, f'fitness_{ckt_name}_gain{desired_gain}_execution{exec_counter}.png'
        )
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  [saved] {save_path}')


# ---------------------------------------------------------------------------
# Summary / comparison plots  (reuses evaluate_sims functions)
# ---------------------------------------------------------------------------

def regenerate_summary_plots(csv_path: str = 'simulations/data.csv',
                              out_dir: str = 'simulations') -> None:
    print('=== Regenerating summary / comparison plots ===')
    simulation_data = data_parse.load_csv_to_json(csv_path)
    grouped = evaluate_sims.group_rows(simulation_data)

    group_summaries = []
    for (ckt_name, desired_gain), rows in sorted(grouped.items()):
        summary = evaluate_sims.summarize_group(rows)
        if summary is not None:
            summary['ckt_name'] = ckt_name
            summary['desired_gain'] = desired_gain
            conv = evaluate_sims.summarize_convergence(rows)
            if conv is not None:
                summary['convergence'] = conv
            group_summaries.append(summary)
        evaluate_sims.plot_group_summary(ckt_name, desired_gain, rows, out_dir=out_dir)

    evaluate_sims.plot_comparison_overview(grouped, group_summaries, out_dir=out_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Regenerate simulation plots from saved data.')
    parser.add_argument('--waveforms', action='store_true', help='Regenerate waveform plots')
    parser.add_argument('--fitness',   action='store_true', help='Regenerate fitness history plots')
    parser.add_argument('--summary',   action='store_true', help='Regenerate summary/comparison plots')
    args = parser.parse_args()

    # Default: regenerate everything
    do_all = not (args.waveforms or args.fitness or args.summary)

    if do_all or args.fitness:
        regenerate_fitness_plots()

    if do_all or args.summary:
        regenerate_summary_plots()

    if do_all or args.waveforms:
        simulation_data = data_parse.load_csv_to_json('simulations/data.csv')
        regenerate_waveforms(simulation_data)

    print('\nDone.')
