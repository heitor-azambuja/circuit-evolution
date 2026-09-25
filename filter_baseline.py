"""
Analytically designed, then quantized, Sallen-Key filters.

This is the control the genetic algorithm is measured against: design each stage
from the textbook equations, round every resistor onto the taps the chosen pots
actually offer, and score the result with exactly the same evaluator the GA
optimises. The gap between this and the GA's solution is the quantization the GA
manages to absorb and the analytical route cannot.

Runs for either resistor realization — one pot per resistor (4R) or two in series
(8R) — since the whole point of the 8R variant is that finer taps should shrink
that gap.
"""
import argparse
import json
import logging
import uuid
from datetime import datetime

import data_parse
import filter_design
import filter_sk4_evolution
import filter_sk8_evolution
from filter_targets import CUTOFF_HZ, ORDER, TARGETS

logger = logging.getLogger()

CIRCUITS = {
    '4r': filter_sk4_evolution,
    '8r': filter_sk8_evolution,
}


def design(module, target: str) -> dict:
    """Ideal and tap-quantized resistors for one target, plus the taps themselves."""
    spec = TARGETS[target]
    caps_nf = spec['capacitors_nf']
    stages = filter_design.stage_parameters(
        ORDER, spec['response'], CUTOFF_HZ, ripple_db=spec['ripple_db'])

    ideal = []
    for (wo, q), (c1_nf, c2_nf) in zip(stages, ((caps_nf[0], caps_nf[1]),
                                                (caps_nf[2], caps_nf[3]))):
        r1, r2 = filter_design.ideal_resistors(wo, q, c1_nf * 1e-9, c2_nf * 1e-9)
        ideal.extend([r1, r2])

    taps = module.quantize(ideal)
    return {
        'stages': stages,
        'ideal_resistors': ideal,
        'taps': taps,
        'quantized_resistors': module.resistor_mapper(taps),
    }


def evaluate(module, target: str, exec_counter: int = 0, auto_plots: bool = False,
             out_dir: str = 'simulations') -> dict:
    """Score the quantized analytical design with the GA's own evaluator."""
    spec = module.SPEC
    evaluator = spec.evaluator
    designed = design(module, target)

    circuit = spec.circuit_factory()
    spec.setup_hook(circuit, target)

    fitness = evaluator.fitness(circuit, designed['quantized_resistors'], target)
    metrics = evaluator.metrics(circuit, designed['quantized_resistors'], target)
    ckt_name = f'{spec.circuit_name}_analytical'

    data_json = {
        'ckt_name': ckt_name,
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
        title = (f'{spec.display_name} — {evaluator.target_label(target)}   '
                 f'(analytical, tap-quantized)')
        evaluator.plot(metrics, f'{out_dir}/{ckt_name}_'
                                f'{evaluator.target_slug(target)}.png', title)

    return data_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plots', action='store_true', default=False)
    parser.add_argument('--target', help='Run only for this target')
    parser.add_argument('--circuit', choices=sorted(CIRCUITS),
                        help='Run only this resistor realization')
    parser.add_argument('--data-csv', default='simulations/filters.csv')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    targets = [args.target] if args.target else list(TARGETS)
    names = [args.circuit] if args.circuit else sorted(CIRCUITS)

    for name in names:
        module = CIRCUITS[name]
        for target in targets:
            designed = design(module, target)
            logger.info(f'{name}/{target}: ideal R = '
                        f'{[round(r, 1) for r in designed["ideal_resistors"]]} Ω')
            logger.info(f'{name}/{target}: quantized R = '
                        f'{[round(r, 1) for r in designed["quantized_resistors"]]} Ω '
                        f'via taps {designed["taps"]}')
            data_json = evaluate(module, target, auto_plots=args.plots)
            logger.info(f'{name}/{target}: fitness = {data_json["solution_fitness"]:.6f}  '
                        f'rmse = {data_json["rmse_db"]:.4f} dB')
            data_parse.dump_json_to_csv(args.data_csv, data_json)


if __name__ == '__main__':
    main()
