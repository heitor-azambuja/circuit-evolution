"""
Frequency-domain objective for the Sallen-Key filter target.

Unlike the amplifier, which is scored on how closely a transient waveform tracks
an amplified copy of its input, a filter is scored on its magnitude response: the
AC sweep's |H(f)| in dB against the curve the requested approximation demands.

The evaluator interface it implements is documented on
`evolution_common.TransientGainEvaluator`.
"""
import numpy as np
import scipy.signal

import filter_design

# Below this level a real implementation is under its own noise floor, so matching
# the ideal curve there is meaningless. Without the floor the deep stopband (where
# a 4th-order target reaches -160 dB at 100 kHz) would dominate a dB-domain error.
_DB_FLOOR = -80.0

# Passband flatness is the requirement most sensitive to component quantization;
# the far stopband's slope is fixed by the filter order, not by the resistor
# values, so weighting it heavily would spend fitness gradient on nothing.
_WEIGHT_PASSBAND = 2.0
_WEIGHT_TRANSITION = 1.0
_WEIGHT_STOPBAND = 0.5


class AcResponseEvaluator:
    """Scores a filter on weighted RMS magnitude error, in dB, against its target curve.

    `targets` maps a target name to its filter specification, e.g.
        {'butterworth': {'response': filter_design.BUTTERWORTH, 'ripple_db': None}}
    """

    def __init__(self, fc_hz: float, order: int, targets: dict,
                 start_hz: float = 10, stop_hz: float = 100_000,
                 points_per_decade: int = 20):
        self.fc_hz = fc_hz
        self.order = order
        self.targets = targets
        self.start_hz = start_hz
        self.stop_hz = stop_hz
        self.points_per_decade = points_per_decade

        # The reference transfer function per target is fixed for the whole run.
        self._coefficients = {}
        for name, spec in targets.items():
            wn = 2 * np.pi * fc_hz
            if spec['response'] == filter_design.BUTTERWORTH:
                self._coefficients[name] = scipy.signal.butter(
                    order, wn, btype='low', analog=True)
            elif spec['response'] == filter_design.CHEBYSHEV:
                self._coefficients[name] = scipy.signal.cheby1(
                    order, spec['ripple_db'], wn, btype='low', analog=True)
            else:
                raise ValueError(f'unknown response for target {name!r}')

    def target_db(self, freqs, target) -> np.ndarray:
        b, a = self._coefficients[target]
        _, h = scipy.signal.freqs(b, a, worN=2 * np.pi * np.asarray(freqs))
        return 20 * np.log10(np.abs(h))

    def weights(self, freqs) -> np.ndarray:
        freqs = np.asarray(freqs)
        w = np.full(freqs.shape, _WEIGHT_STOPBAND)
        w[freqs <= 10 * self.fc_hz] = _WEIGHT_TRANSITION
        w[freqs <= self.fc_hz] = _WEIGHT_PASSBAND
        return w

    def weighted_rmse_db(self, freqs, response_db, target) -> float:
        reference = self.target_db(freqs, target)
        error = np.maximum(response_db, _DB_FLOOR) - np.maximum(reference, _DB_FLOOR)
        w = self.weights(freqs)
        return float(np.sqrt(np.sum(w * error ** 2) / np.sum(w)))

    def _sweep(self, circuit, resistances) -> tuple:
        circuit.configure_resistors(resistances)
        analysis = circuit.ac_analysis(start_frequency=self.start_hz,
                                       stop_frequency=self.stop_hz,
                                       points_per_decade=self.points_per_decade)
        freqs = np.array(analysis.frequency)
        response_db = 20 * np.log10(np.abs(np.array(analysis['out'])))
        return freqs, response_db

    def fitness(self, circuit, resistances, target) -> float:
        freqs, response_db = self._sweep(circuit, resistances)
        rmse = self.weighted_rmse_db(freqs, response_db, target)
        # Bounded in (0, 1], monotonic in the error, and needs no special case for
        # a perfect match — unlike the amplifier's unbounded 1/euclidean.
        return 1.0 / (1.0 + rmse)

    def metrics(self, circuit, resistances, target) -> dict:
        freqs, response_db = self._sweep(circuit, resistances)
        reference_db = self.target_db(freqs, target)
        passband = freqs <= self.fc_hz

        return {
            'resistances': list(resistances),
            'freqs': freqs,
            'response_db': response_db,
            'target_db': reference_db,
            'rmse_db': self.weighted_rmse_db(freqs, response_db, target),
            'max_passband_error_db': float(np.max(np.abs(
                response_db[passband] - reference_db[passband]))),
            'cutoff_realized_hz': _minus_3db_crossing(freqs, response_db),
            # Interpolated in log-frequency: the sweep grid is logarithmic, and
            # 10*fc only lands exactly on it for particular sweep parameters.
            'attenuation_at_10fc_db': float(np.interp(
                np.log10(10 * self.fc_hz), np.log10(freqs), response_db)),
        }

    def plot(self, metrics: dict, save_path: str, title: str) -> None:
        _plot_response(metrics, save_path, title, self.fc_hz)

    def target_fields(self, target) -> dict:
        return {
            'response': target,
            'order': self.order,
            'cutoff_hz': self.fc_hz,
            'ripple_db': self.targets[target].get('ripple_db'),
        }

    def metric_fields(self, metrics: dict) -> dict:
        fields = {key: metrics[key] for key in (
            'rmse_db', 'max_passband_error_db', 'cutoff_realized_hz',
            'attenuation_at_10fc_db')}
        fields['cutoff_error_percent'] = (
            (metrics['cutoff_realized_hz'] - self.fc_hz) / self.fc_hz * 100
            if metrics['cutoff_realized_hz'] is not None else None
        )
        fields['resistors'] = metrics['resistances']
        return fields

    def target_slug(self, target) -> str:
        return f'{target}{self.order}p'

    def target_label(self, target) -> str:
        return f'{target.capitalize()} {self.order}th order, fc = {self.fc_hz:.0f} Hz'

    def target_from_row(self, row: dict):
        """Recover the target from a saved CSV row, for regenerating plots."""
        raw = row.get('response')
        return raw if raw in self.targets else None


def _minus_3db_crossing(freqs, response_db):
    """Frequency where the response first falls through -3 dB, log-interpolated."""
    below = np.flatnonzero(response_db <= -3.0)
    if below.size == 0 or below[0] == 0:
        return None
    i = below[0]
    f_lo, f_hi = np.log10(freqs[i - 1]), np.log10(freqs[i])
    db_lo, db_hi = response_db[i - 1], response_db[i]
    if db_hi == db_lo:
        return float(freqs[i])
    return float(10 ** (f_lo + (-3.0 - db_lo) * (f_hi - f_lo) / (db_hi - db_lo)))


def _plot_response(metrics: dict, save_path: str, title: str, fc_hz: float) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 7),
                             gridspec_kw={'height_ratios': [3, 1]},
                             constrained_layout=True, sharex=True)

    freqs = metrics['freqs']
    response = metrics['response_db']
    reference = metrics['target_db']

    ax = axes[0]
    ax.semilogx(freqs, reference, color='#2ca02c', linewidth=1.8, linestyle='--',
                label='target', zorder=3)
    ax.semilogx(freqs, response, color='#1f77b4', linewidth=1.8, label='obtained', zorder=4)
    ax.axvline(fc_hz, color='#888888', linewidth=1.0, linestyle=':')
    ax.set_ylabel('|H(f)| [dB]', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylim(max(_DB_FLOOR, np.min(reference) - 5), 5)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, which='both', linestyle=':', alpha=0.6)

    res_str = ', '.join(f'{r/1000:.2f}k' for r in metrics['resistances'])
    realized = metrics['cutoff_realized_hz']
    lines = [
        f"weighted rmse: {metrics['rmse_db']:.3f} dB",
        f"max passband err: {metrics['max_passband_error_db']:.3f} dB",
        f'fc realized: {realized:.1f} Hz' if realized is not None
        else 'fc realized: not reached',
        f'R: [{res_str}] Ω',
    ]
    ax.text(0.01, 0.06, '\n'.join(lines), transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85,
                      edgecolor='#aaaaaa'))

    axe = axes[1]
    error = np.maximum(response, _DB_FLOOR) - np.maximum(reference, _DB_FLOOR)
    axe.axvspan(freqs[0], fc_hz, color='#1f77b4', alpha=0.08)
    axe.axvspan(fc_hz, min(10 * fc_hz, freqs[-1]), color='#ff7f0e', alpha=0.08)
    axe.semilogx(freqs, error, color='#d62728', linewidth=1.2)
    axe.axhline(0, color='#555555', linewidth=0.8)
    axe.set_xlabel('frequency [Hz]', fontsize=11)
    axe.set_ylabel('error [dB]', fontsize=11)
    axe.grid(True, which='both', linestyle=':', alpha=0.6)

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
