"""
Evolve the resistor taps of a 4th-order Sallen-Key low-pass filter.

Same approach as the amplifier scripts — a GA picks digital-potentiometer tap
positions — but the objective is the magnitude response, not a gain.

Capacitors are fixed hardware and differ per target and per stage. That is forced,
not arbitrary: a unity-gain stage reaches at most Q = 0.5*sqrt(C1/C2), and the two
stages of a 4th-order cascade need quite different Q (0.54 and 1.31 for
Butterworth, 0.71 and 2.94 for a 0.5 dB Chebyshev). Each pair below was chosen so
the ideal resistors land near mid-scale on a 10k pot, where the 101 Ω tap step is
about 2% — the quantization the GA has to work around.
"""
import logging

import circuits
import components
import filter_design
from evolution_common import CircuitSpec, run_cli
from filter_evaluation import AcResponseEvaluator

logger = logging.getLogger()
logger.setLevel(logging.INFO)

CUTOFF_HZ = 1000.0
ORDER = 4

# capacitors_nf is (C1, C2) of the low-Q stage then (C1, C2) of the high-Q stage.
TARGETS = {
    'butterworth': {
        'response': filter_design.BUTTERWORTH,
        'ripple_db': None,
        'capacitors_nf': (27.0, 22.0, 82.0, 10.0),
    },
    'chebyshev': {
        'response': filter_design.CHEBYSHEV,
        'ripple_db': 0.5,
        'capacitors_nf': (56.0, 27.0, 150.0, 3.9),
    },
}

pot_10k = components.DigitalPot(10000, 100)


def resistor_mapper(solution) -> list:
    """Order: R1a, R2a (low-Q stage), R1b, R2b (high-Q stage) — all on one 10k pot."""
    return [pot_10k.get_resistance(value) for value in solution]


def configure(circuit, target) -> None:
    circuit.configure_capacitors(*TARGETS[target]['capacitors_nf'])


SPEC = CircuitSpec(
    circuit_name='sallen_key_lp_4p',
    display_name='Sallen-Key Low-Pass',
    circuit_factory=circuits.SallenKeyLowPass,
    num_genes=4,
    resistor_mapper=resistor_mapper,
    evaluator=AcResponseEvaluator(fc_hz=CUTOFF_HZ, order=ORDER, targets=TARGETS),
    setup_hook=configure,
    default_population=20,
    default_generations=400,
)


if __name__ == "__main__":
    run_cli(SPEC, target_list=list(TARGETS), data_csv='simulations/filters.csv')
