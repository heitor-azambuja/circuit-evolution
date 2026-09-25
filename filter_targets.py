"""
The filter specification shared by every Sallen-Key run script.

Capacitors are fixed hardware chosen per target and per stage, and are the same
for the 4R and 8R variants on purpose: holding them constant is what makes the
two variants a clean comparison of *resistor resolution* and nothing else.

Why they must differ per stage and per target: a unity-gain Sallen-Key stage can
only reach Q = 0.5*sqrt(C1/C2), and the two stages of a 4th-order cascade need
quite different Q (0.54 and 1.31 for Butterworth, 0.71 and 2.94 for a 0.5 dB
Chebyshev). Each pair below was picked so the ideal resistors land near mid-scale
on a 10 kΩ pot. See filter_design.max_q.
"""
import filter_design

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


def configure(circuit, target) -> None:
    circuit.configure_capacitors(*TARGETS[target]['capacitors_nf'])
