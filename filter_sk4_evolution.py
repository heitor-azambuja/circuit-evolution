"""
Evolve a 4th-order Sallen-Key low-pass filter, one X9C103 pot per resistor.

Same approach as the amplifier scripts — a GA picks digital-potentiometer tap
positions — but the objective is the magnitude response, not a gain.

Each of the four resistors is a single X9C103 (10 kΩ, 100 taps), so the step is
101 Ω. The ideal resistors sit near 5 kΩ, where that is about 2% — the
quantization the GA has to work around. `filter_sk8_evolution` is the same filter
with a finer realization, for comparison.
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

pot_10k = components.DigitalPot(10000, 100)   # X9C103


def resistor_mapper(solution) -> list:
    """Order: R1a, R2a (low-Q stage), R1b, R2b (high-Q stage)."""
    return [pot_10k.get_resistance(value) for value in solution]


def quantize(resistances) -> list:
    """Nearest taps for a set of ideal resistances — the analytical baseline's genes."""
    return filter_design.quantize_to_pot(resistances, pot_10k)


SPEC = CircuitSpec(
    circuit_name='sallen_key_lp_4p',
    display_name='Sallen-Key Low-Pass (4R)',
    circuit_factory=circuits.SallenKeyLowPass,
    num_genes=4,
    resistor_mapper=resistor_mapper,
    evaluator=AcResponseEvaluator(order=ORDER, targets=TARGETS),
    setup_hook=configure,
    default_population=20,
    default_generations=400,
)


if __name__ == "__main__":
    run_cli(SPEC, target_list=list(TARGETS), data_csv='simulations/filters.csv')
