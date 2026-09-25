"""
Analytical design equations for the unity-gain Sallen-Key low-pass filter.

Pure math — no SPICE, no simulation. Two jobs:

1. Turn a filter specification (order, approximation, cutoff) into the per-stage
   (wo, Q) pairs a cascade has to realise.
2. Turn a (wo, Q) pair into the ideal resistor values, and quantize those onto a
   digital potentiometer's taps.

(2) is what produces the *analytically designed, then quantized* baseline that
the genetic algorithm is compared against.

Transfer function of one unity-gain stage, with R1/R2 in series from the input
and C1 bootstrapped from the mid-node to the output:

    H(s) = 1 / (1 + s*C2*(R1 + R2) + s^2*R1*R2*C1*C2)

so, against the canonical 1 / (1 + s/(Q*wo) + (s/wo)^2):

    wo = 1 / sqrt(R1*R2*C1*C2)
    Q  = sqrt(R1*R2*C1/C2) / (R1 + R2)

Two consequences drive the whole design:

* Q is maximised at R1 == R2, where it equals 0.5*sqrt(C1/C2). So with *equal*
  capacitors Q can never exceed 0.5, and a 4th-order Butterworth (which needs
  Q = 1.306) is unreachable by tuning resistors alone. The capacitors must be
  deliberately unequal — see `max_q`.
* H(s) depends on R1 and R2 only through their sum and product, so swapping them
  leaves the response identical. Every stage therefore has two equivalent
  resistor assignments, and the GA's search space is symmetric.
"""
import numpy as np
import scipy.signal

BUTTERWORTH = 'butterworth'
CHEBYSHEV = 'chebyshev'


def stage_parameters(order: int, response: str, fc_hz: float,
                     ripple_db: float = None) -> list:
    """Return [(wo_rad_s, q), ...] for each 2nd-order section of the cascade.

    Stages come back ordered by ascending Q, the usual cascade convention: the
    low-Q section sits first so the high-Q section's peaking is applied to an
    already band-limited signal.
    """
    if order % 2 != 0:
        raise ValueError(f'order must be even (cascade of 2nd-order stages), got {order}')

    wn = 2 * np.pi * fc_hz
    if response == BUTTERWORTH:
        _, poles, _ = scipy.signal.butter(order, wn, btype='low', analog=True, output='zpk')
    elif response == CHEBYSHEV:
        if ripple_db is None:
            raise ValueError('ripple_db is required for a Chebyshev response')
        _, poles, _ = scipy.signal.cheby1(order, ripple_db, wn, btype='low',
                                          analog=True, output='zpk')
    else:
        raise ValueError(f'unknown response {response!r}')

    stages = []
    for pole in poles:
        if pole.imag <= 0:
            continue  # take one pole per conjugate pair
        wo = float(np.abs(pole))
        stages.append((wo, wo / (2 * abs(float(pole.real)))))

    return sorted(stages, key=lambda s: s[1])


def max_q(c1: float, c2: float) -> float:
    """Highest Q a unity-gain stage can reach with these capacitors (at R1 == R2)."""
    return 0.5 * np.sqrt(c1 / c2)


def ideal_resistors(wo: float, q: float, c1: float, c2: float) -> tuple:
    """Exact resistors realising (wo, Q) with capacitors c1, c2. Returns (larger, smaller).

    The two values are interchangeable in the circuit; the order is only for
    deterministic output.
    """
    reachable = max_q(c1, c2)
    if q > reachable:
        raise ValueError(
            f'Q={q:.4f} is unreachable with C1={c1:.3e} F, C2={c2:.3e} F '
            f'(max Q = 0.5*sqrt(C1/C2) = {reachable:.4f}). Increase the C1/C2 ratio.'
        )

    product = 1.0 / (wo ** 2 * c1 * c2)
    total = np.sqrt(product * c1 / c2) / q
    # At q == max_q the discriminant is analytically zero; floating point lands
    # just below it, so clamp rather than take the square root of -1e-22.
    spread = np.sqrt(max(0.0, total ** 2 - 4 * product))
    return (total + spread) / 2, (total - spread) / 2


def transfer_function(r1: float, r2: float, c1: float, c2: float) -> tuple:
    """Numerator/denominator coefficients of one stage, for scipy.signal."""
    return [1.0], [r1 * r2 * c1 * c2, c2 * (r1 + r2), 1.0]


def quantize_to_pot(values, pot) -> list:
    """Round resistances onto a DigitalPot's nearest tap positions."""
    taps = []
    for value in values:
        tap = int(round(value / pot.min_value)) + 1
        taps.append(max(1, min(pot.tap_points, tap)))
    return taps


def quantize_to_series_pots(values, coarse, fine) -> list:
    """Nearest (coarse tap, fine tap) pair per value, for two pots wired in series.

    Returned interleaved as [coarse, fine, coarse, fine, ...] to match the gene
    order the 8-gene mappers use. Searched exhaustively: the tap counts are small,
    and an exact answer is worth more here than a closed form that has to reason
    about where the two step sizes interleave.
    """
    taps = []
    for value in values:
        best = None
        for c in range(1, coarse.tap_points + 1):
            base = coarse.get_resistance(c)
            for f in range(1, fine.tap_points + 1):
                error = abs(base + fine.get_resistance(f) - value)
                if best is None or error < best[0]:
                    best = (error, c, f)
        taps.extend([best[1], best[2]])
    return taps
