"""
Analytical design equations for the BJT class-A amplifier.

The counterpart of `filter_design`, and the contrast with it is the point. The
filter's equations are exact: give them R and C and the transfer function follows,
so the only error left is quantization. These are not. They are small-signal
approximations to a nonlinear device, and they miss the target gain by about ten
percent even with resistors of infinite precision -- `re' = VT/IC` ignores base
resistance and the Early effect, and the operating point comes from an *assumed*
V_BE. Since the gain is proportional to IC, an error in V_BE passes straight
through.

Design criterion is maximum output swing, V_C = V_CC/2. That fixes IC once Rc is
chosen and the gain requirement then fixes Rc, so there is no free parameter left
to quietly tune the baseline with:

    Av = (Rc||RL)/re',  re' = VT/IC,  Rc*IC = VCC/2
    =>  Rc = RL * (VCC/(2*VT*Av) - 1)

The base divider is the part that does not survive contact with the hardware. The
textbook rule sizes it from the base current (I_div ~ 10*I_B), which at these
collector currents asks for megohms -- more than an order of magnitude past what
a 100k potentiometer can reach. `divider` therefore sizes it to the pot instead,
which is a concession in the analytical design's favour, and `textbook_divider_total`
reports what the rule would have asked for so the gap can be quoted.
"""
import numpy as np

# Thermal voltage at 25 C, the temperature the simulations run at.
VT_25C = 0.02585

VCC = 3.3
LOAD = 10_000.0
VBE = 0.6
VE_FRACTION = 0.1


def collector_resistor(gain: float, vcc: float = VCC, load: float = LOAD,
                       vt: float = VT_25C) -> float:
    """Rc for a target gain under the maximum-swing criterion."""
    resistance = load * (vcc / (2 * vt * gain) - 1)
    if resistance <= 0:
        raise ValueError(
            f'gain {gain} needs Rc <= 0 with VCC={vcc} V and RL={load} ohm; the '
            f'maximum swing criterion cannot reach it')
    return resistance


def design(gain: float, vcc: float = VCC, load: float = LOAD, vt: float = VT_25C,
           vbe: float = VBE, ve_fraction: float = VE_FRACTION,
           divider_total: float = None) -> dict:
    """Ideal resistances in the circuit's order: R1, R2, Rc, Re.

    `divider_total` is R1+R2. Left as None it is not decided here — call
    `divider` with the potentiometer to size it.
    """
    rc = collector_resistor(gain, vcc, load, vt)
    collector_current = (vcc / 2) / rc
    emitter_voltage = ve_fraction * vcc
    re = emitter_voltage / collector_current

    base_voltage = emitter_voltage + vbe
    ratio = base_voltage / vcc
    if divider_total is None:
        r1 = r2 = float('nan')
    else:
        r1, r2 = divider_total * (1 - ratio), divider_total * ratio

    return {
        'resistances': [r1, r2, rc, re],
        'collector_current': collector_current,
        'emitter_resistance_small_signal': vt / collector_current,
        'base_voltage': base_voltage,
        'divider_ratio': ratio,
    }


def divider(base_voltage: float, pot, vcc: float = VCC) -> tuple:
    """Largest R1, R2 setting this base voltage that the pot can still realise.

    R1 is the larger leg, so it is what bumps into the pot's full scale first.
    """
    ratio = base_voltage / vcc
    total = pot.max_value / (1 - ratio)
    return total * (1 - ratio), total * ratio


def textbook_divider_total(collector_current: float, beta: float, vcc: float = VCC,
                           stiffness: float = 10.0) -> float:
    """R1+R2 the classic rule asks for: a divider current `stiffness` times I_B."""
    return vcc / (stiffness * collector_current / beta)


def quantize(resistances, pot) -> list:
    """Nearest tap per resistance, saturating at both ends."""
    taps = []
    for value in resistances:
        taps.append(max(1, min(pot.tap_points, int(round(value / pot.min_value)) + 1)))
    return taps


def exceeds_pot(resistances, pot) -> list:
    """Which resistances the pot cannot reach at all — these saturate when quantized."""
    return [not (0 <= value <= pot.max_value) for value in resistances]
