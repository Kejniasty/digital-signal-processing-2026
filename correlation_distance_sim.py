from dsp_signal import Signal
from dsp_signal import cross_correlate_direct
import math


def _generate_probe_signal(t: float, period: float) -> float:
    """
    Composite periodic probing signal built from two basic sinusoids
    (as required by the spec: at least two basic periodic signals).
    """
    return (math.sin(2 * math.pi * t / period) +
            0.5 * math.sin(4 * math.pi * t / period))


class DistanceSensor:
    """
    Simulated correlation-based distance sensor.

    Parameters
    ----------
    signal_speed    : float  – speed of the probing signal in the medium  [units/s]
    object_speed    : float  – speed of the tracked object (constant)     [units/s]
    initial_distance: float  – starting distance of the object            [units]
    sample_rate     : int    – sampling frequency                          [Hz]
    buffer_size     : int    – number of discrete samples per buffer
    probe_period    : float  – period of the continuous probing signal     [s]
    report_interval : float  – how often the sensor updates its reading   [s]
    time_unit       : float  – basic simulation time step                  [s]
    """

    def __init__(self,
                 signal_speed: float = 340.0,
                 object_speed: float = 10.0,
                 initial_distance: float = 500.0,
                 sample_rate: int = 1000,
                 buffer_size: int = 256,
                 probe_period: float = 0.1,
                 report_interval: float = 0.05,
                 time_unit: float = 1e-4):

        self.signal_speed = signal_speed
        self.object_speed = object_speed
        self.distance = initial_distance
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.probe_period = probe_period
        self.report_interval = report_interval
        self.time_unit = time_unit

        self._dt = 1.0 / sample_rate  # sampling period [s]
        self._sim_time = 0.0  # current simulation time [s]
        self._last_report = 0.0

        # circular sample buffers
        self._probe_buf: list[float] = []
        self._return_buf: list[float] = []

        # log: (sim_time, true_distance, estimated_distance)
        self.log: list[tuple[float, float, float]] = []

    # ── helpers ──────────────────────────────────────────────────────────────

    def _true_distance(self, t: float) -> float:
        """Object moves away at constant speed."""
        return self.distance + self.object_speed * t

    def _probe_sample(self, t: float) -> float:
        return _generate_probe_signal(t, self.probe_period)

    def _return_sample(self, t: float) -> float:
        """Return signal = probe signal delayed by 2d/v (round-trip time)."""
        d = self._true_distance(t)
        tau = 2.0 * d / self.signal_speed  # round-trip delay [s]
        return _generate_probe_signal(t - tau, self.probe_period)

    # ── estimation ───────────────────────────────────────────────────────────

    def _estimate_distance(self) -> float:
        """
        Steps 3-7 from the spec:
        Compute cross-correlation of the two buffers, find the peak in the
        right half, convert sample-lag to time, then to distance.
        """
        if len(self._probe_buf) < self.buffer_size:
            return float('nan')

        h_sig = Signal(self._probe_buf[-self.buffer_size:],
                       1.0, self.buffer_size * self._dt,
                       0.0, 0.0, self.sample_rate)
        x_sig = Signal(self._return_buf[-self.buffer_size:],
                       1.0, self.buffer_size * self._dt,
                       0.0, 0.0, self.sample_rate)

        corr = cross_correlate_direct(h_sig, x_sig)
        vals = corr.signal

        # Right half: index > centre  (positive lags → return arrives after probe)
        centre = len(vals) // 2
        right = vals[centre:]
        if not right:
            return float('nan')

        peak_idx = right.index(max(right))  # lag in samples (from centre)
        lag_samples = peak_idx  # number of samples of delay
        delay_time = lag_samples * self._dt  # delay in seconds
        distance = (delay_time * self.signal_speed) / 2.0
        return distance

    # ── main simulation loop ──────────────────────────────────────────────────

    def run(self, total_time: float) -> list[tuple[float, float, float]]:
        """
        Simulate sensor operation for `total_time` seconds.

        Returns list of (sim_time, true_distance, estimated_distance) tuples
        recorded at each report interval.
        """
        steps = int(total_time / self.time_unit)
        sample_counter = 0
        report_counter = 0

        for step in range(steps):
            t = step * self.time_unit

            # Accumulate discrete samples at the given sample_rate
            if t >= sample_counter * self._dt:
                self._probe_buf.append(self._probe_sample(t))
                self._return_buf.append(self._return_sample(t))
                sample_counter += 1

            # Report at report_interval
            if t >= report_counter * self.report_interval:
                est = self._estimate_distance()
                true = self._true_distance(t)
                self.log.append((t, true, est))
                report_counter += 1

        return self.log