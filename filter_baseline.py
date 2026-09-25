"""
Analytically designed, then quantized, Sallen-Key filters.

This is the control the genetic algorithm is measured against: design each stage
from the textbook equations, round every resistor to its nearest digital-pot tap,
and score the result with exactly the same evaluator the GA optimises. The gap
between this and the GA's solution is the quantization the GA manages to absorb
and the analytical route cannot.

Run it after (or before) the GA; it appends rows to the same CSV with a distinct
ckt_name, so both live in one table.
"""
import argparse
import json
import logging
import uuid
from datetime import datetime

import data_parse
import filter_design
import filter_sk4_evolution as sk4

logger = logging.getLogger()

BASELINE_CKT_NAME = 'sallen_key_lp_4p_analytical'


def design(target: str) -> dict:
    """Ideal and tap-quantized resistors for one target, plus the taps themselves."""
    spec = sk4.TARGETS[target]
    caps_nf = spec['capacitors_nf']
    stages = filter_design.stage_parameters(
        sk4.ORDER, spec['response'], sk4.CUTOFF_HZ, ripple_db=spec['ripple_db'])

    ideal = []
    for (wo, q), (c1_nf, c2_nf) in zip(stages, ((caps_nf[0], caps_nf[1]),
                                                (caps_nf[2], caps_nf[3]))):
        r1, r2 = filter_design.ideal_resistors(wo, q, c1_nf * 1e-9, c2_nf * 1e-9)
        ideal.extend([r1, r2])

    taps = filter_design.quantize_to_pot(ideal, sk4.pot_10k)
    return {
        'stages': stages,
        'ideal_resistors': ideal,
        'taps': taps,
        'quantized_resistors': sk4.resistor_mapper(taps),
    }


def evaluate(target: str, exec_counter: int = 0, auto_plots: bool = False,
             out_dir: str = 'simulations') -> dict:
    """Score the quantized analytical design with the GA's own evaluator."""
    evaluator = sk4.SPEC.evaluator
    designed = design(target)

    circuit = sk4.SPEC.circuit_factory()
    sk4.SPEC.setup_hook(circuit, target)

    fitness = evaluator.fitness(circuit, designed['quantized_resistors'], target)
    metrics = evaluator.metrics(circuit, designed['quantized_resistors'], target)

    data_json = {
        'ckt_name': BASELINE_CKT_NAME,
        **evaluator.target_fields(target),
        'exec_counter': exec_counter,
        'seed': None,
        'run_id': str(uuid.uuid4()),
        'timestamp': datetime.now().isoformat(),
        'solution': json.dumps(designed['taps']),
        'solution_fitness': fitness,
        **evaluator.metric_fields(metrics),
        'ideal_resistors': json.dumps([round(r, 3) for r in designed['ideal_resistors']]),
    }

    if auto_plots:
        title = (f'{sk4.SPEC.display_name} — {evaluator.target_label(target)}   '
                 f'(analytical, tap-quantized)')
        evaluator.plot(metrics, f'{out_dir}/{BASELINE_CKT_NAME}_'
                                f'{evaluator.target_slug(target)}.png', title)

    return data_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plots', action='store_true', default=False)
    parser.add_argument('--target', help='Run only for this target')
    parser.add_argument('--data-csv', default='simulations/filters.csv')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    targets = [args.target] if args.target else list(sk4.TARGETS)

    for target in targets:
        designed = design(target)
        logger.info(f'{target}: ideal R = '
                    f'{[round(r, 1) for r in designed["ideal_resistors"]]} Ω '
                    f'-> taps {designed["taps"]}')
        data_json = evaluate(target, auto_plots=args.plots)
        logger.info(f'{target}: quantized fitness = {data_json["solution_fitness"]:.6f}  '
                    f'rmse = {data_json["rmse_db"]:.4f} dB')
        data_parse.dump_json_to_csv(args.data_csv, data_json)


if __name__ == '__main__':
    main()
