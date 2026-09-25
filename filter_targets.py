"""
The filter specification shared by every Sallen-Key run script.

A target is one (response family, cutoff) pair. Cutoff is the continuous
specification the experiment sweeps, mirroring the amplifier's four gains: the
same board, retuned by software, which is the adaptability claim itself.

Capacitors are fixed hardware and are the same across every target and both the
4R and 8R variants. That is deliberate — holding them constant is what makes the
sweep a clean test of resistor resolution and of how the analytical design
degrades, rather than a comparison between different boards.

Why they must differ per stage: a unity-gain Sallen-Key stage can only reach
Q = 0.5*sqrt(C1/C2), and the two stages of a 4th-order cascade need quite
different Q (0.54 and 1.31 for Butterworth, 0.71 and 2.94 for a 0.5 dB
Chebyshev). See filter_design.max_q.

Because the resistors scale as 1/fc with the capacitors fixed, the cutoffs below
are bounded by the potentiometer at both ends: under about 900 Hz the ideal
resistors exceed a 10k pot, and above about 3 kHz they fall into its coarsest
taps. Resolution therefore degrades as the cutoff rises, which is the trend the
experiment is built to measure.
"""
import filter_design

ORDER = 4

# Sweep of the continuous specification. Bounded by the pot at both ends.
CUTOFFS_HZ = (1000.0, 1500.0, 2000.0, 3000.0)

# capacitors_nf is (C1, C2) of the low-Q stage then (C1, C2) of the high-Q stage.
RESPONSES = {
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

TARGETS = {
    f'{family}_{int(cutoff)}': dict(spec, cutoff_hz=cutoff)
    for family, spec in RESPONSES.items()
    for cutoff in CUTOFFS_HZ
}


def configure(circuit, target) -> None:
    circuit.configure_capacitors(*TARGETS[target]['capacitors_nf'])
