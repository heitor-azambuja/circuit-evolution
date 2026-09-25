import logging

import circuits
import components
from evolution_common import CircuitSpec, run_cli

logger = logging.getLogger()
logger.setLevel(logging.INFO)

pot_100k = components.DigitalPot(100000, 100)


def resistor_mapper(solution) -> list:
    """Order: R1, R2, Rc, Re — all on the same 100k pot."""
    return [pot_100k.get_resistance(value) for value in solution]


def configure(circuit, target) -> None:
    """Coupling/bypass capacitors in µF; independent of the gain being targeted."""
    circuit.configure_capacitors(47, 100, 47)


SPEC = CircuitSpec(
    circuit_name='bjt_class_a_amp_4r',
    display_name='BJT Class A Amplifier',
    circuit_factory=circuits.BJTClassAAmp,
    num_genes=4,
    resistor_mapper=resistor_mapper,
    setup_hook=configure,
    default_population=20,
    default_generations=400,
)


if __name__ == "__main__":
    run_cli(SPEC)
