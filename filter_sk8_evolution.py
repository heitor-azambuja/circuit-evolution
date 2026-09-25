"""
Evolve the same 4th-order Sallen-Key low-pass filter, two pots in series per resistor.

Each of the four filter resistors is an X9C103 (10 kΩ) in series with an X9C102
(1 kΩ), so the effective step is 10.1 Ω instead of 101 Ω — ten times finer around
the ~5 kΩ the design calls for — over a 0..11 kΩ range. Eight genes instead of
four, mirroring how amp_8r refines amp_4r.

The capacitors and the target curves are identical to the 4R variant, so the two
differ only in resistor resolution. That is the comparison: does a second pot per
resistor, which costs board area and pins, buy enough accuracy to matter — and
does it erase the advantage the GA has over an analytically designed, quantized
solution?
"""
import logging

import circuits
import components
import filter_design
from evolution_common import CircuitSpec, run_cli
from filter_evaluation import AcResponseEvaluator
from filter_targets import ORDER, TARGETS, configure

logger = logging.getLogger()
logger.setLevel(logging.INFO)

pot_10k = components.DigitalPot(10000, 100)   # X9C103, coarse
pot_1k = components.DigitalPot(1000, 100)     # X9C102, fine


def resistor_mapper(solution) -> list:
    """Order: coarse/fine pairs for R1a, R2a, R1b, R2b — each resistor is a series pair."""
    return [pot_10k.get_resistance(solution[i]) + pot_1k.get_resistance(solution[i + 1])
            for i in range(0, 8, 2)]


def quantize(resistances) -> list:
    """Nearest coarse/fine tap pairs — the analytical baseline's genes."""
    return filter_design.quantize_to_series_pots(resistances, pot_10k, pot_1k)


SPEC = CircuitSpec(
    circuit_name='sallen_key_lp_4p_8r',
    display_name='Sallen-Key Low-Pass (8R)',
    circuit_factory=circuits.SallenKeyLowPass,
    num_genes=8,
    resistor_mapper=resistor_mapper,
    evaluator=AcResponseEvaluator(order=ORDER, targets=TARGETS),
    setup_hook=configure,
    default_population=40,
    default_generations=400,
)


if __name__ == "__main__":
    run_cli(SPEC, target_list=list(TARGETS), data_csv='simulations/filters.csv')
