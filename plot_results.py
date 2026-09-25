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

import data_parse
import evaluate_sims


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _specs() -> dict:
    """Every known circuit, keyed by the ckt_name written into the CSV.

    The run scripts' own SPECs are the registry, so a circuit can never be
    described in two places — and an unknown ckt_name is skipped rather than
    silently plotted as some other circuit.
    """
    import amp_4r_evolution
    import amp_8r_evolution
    import filter_sk4_evolution

    return {spec.circuit_name: spec for spec in (
        amp_4r_evolution.SPEC,
        amp_8r_evolution.SPEC,
        filter_sk4_evolution.SPEC,
    )}


def _build_circuit(spec, target):
    circuit = spec.circuit_factory()
    if spec.setup_hook is not None:
        spec.setup_hook(circuit, target)
    return circuit


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


# ---------------------------------------------------------------------------
# Waveform plots
# ---------------------------------------------------------------------------

def regenerate_waveforms(rows: list, out_dir: str = 'simulations') -> None:
    print('=== Regenerating result plots ===')
    specs = _specs()

    for row in rows:
        ckt_name     = row.get('ckt_name', '')
        exec_counter = row.get('exec_counter', '?')
        seed         = row.get('seed', '')

        spec = specs.get(ckt_name)
        if spec is None:
            print(f'  [skip] exec={exec_counter} — unknown circuit {ckt_name!r}')
            continue

        evaluator = spec.evaluator
        target = evaluator.target_from_row(row)
        resistances = _get_resistances(row, ckt_name)
        if resistances is None or target is None:
            print(f'  [skip] exec={exec_counter} — missing resistors or target')
            continue

        try:
            circuit = _build_circuit(spec, target)
            metrics = evaluator.metrics(circuit, resistances, target)
        except Exception as exc:
            print(f'  [error] exec={exec_counter}: {exc}')
            continue

        title = (
            f'{spec.display_name} — {evaluator.target_label(target)}   '
            f'(exec {exec_counter}, seed {seed})'
        )
        path = os.path.join(
            out_dir,
            f'{ckt_name}_{evaluator.target_slug(target)}_execution{exec_counter}.png')
        evaluator.plot(metrics, path, title)
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
        exec_counter = data.get('exec_counter', '?')
        # Runs before the filter target only recorded the gain.
        target_slug = data.get('target_slug') or f"gain{data.get('desired_gain', '?')}"

        if not history:
            continue

        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        ax.plot(history, linewidth=2, color='#1f77b4')
        ax.set_xlabel('Generation', fontsize=11)
        ax.set_ylabel('Fitness', fontsize=11)
        ax.set_title(
            f'Fitness — {ckt_name} {target_slug} exec={exec_counter}',
            fontsize=11
        )
        ax.grid(True, linestyle=':', alpha=0.6)

        save_path = os.path.join(
            out_dir, f'fitness_{ckt_name}_{target_slug}_execution{exec_counter}.png'
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
