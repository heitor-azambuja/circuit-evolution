import logging

import circuits
import components
from evolution_common import CircuitSpec, run_cli

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# An X9C104 in series with an X9C103 per resistor: eight parts.
POT_SPECS = [(100000, 100), (10000, 100)] * 4
pot_100k = components.DigitalPot(*POT_SPECS[0])
pot_10k = components.DigitalPot(*POT_SPECS[1])


def resistor_mapper(solution) -> list:
    """Order: R11, R12, R21, R22, Rc1, Rc2, Re1, Re2 — alternating 100k/10k pots."""
    resistances = []
    for i in range(0, 7, 2):
        resistances.append(pot_100k.get_resistance(solution[i]))
        resistances.append(pot_10k.get_resistance(solution[i + 1]))
    return resistances


def configure(circuit, target) -> None:
    """Coupling/bypass capacitors in µF; independent of the gain being targeted."""
    circuit.configure_capacitors(47, 100, 47)


SPEC = CircuitSpec(
    circuit_name='bjt_class_a_amp_8r',
    display_name='BJT Class A Amplifier (8R)',
    circuit_factory=circuits.BJTClassAAmp8R,
    num_genes=8,
    resistor_mapper=resistor_mapper,
    setup_hook=configure,
    default_population=40,
    default_generations=400,
)


if __name__ == "__main__":
    run_cli(SPEC)
