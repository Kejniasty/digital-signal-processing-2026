import math
import cmath
import random
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum


class Signal:
    """Represents a continuous or discrete signal."""

    def __init__(self, signal=None, amplitude=0.0, duration=0.0,
                 start_time=0.0, period=0.0, sample_rate=100, is_sampled=False):
        self.signal = signal.copy() if signal is not None else []
        self.amplitude = amplitude
        self.duration = duration
        self.start_time = start_time
        self.period = period
        self.sample_rate = sample_rate
        self.is_sampled = is_sampled

        # Generate time axis
        self.time = [
            start_time + i / sample_rate
            for i in range(len(self.signal))
        ]

    def pad(self, other: "Signal"):
        """Pad the shorter signal with zeros so both have equal length."""
        self_signal = self.signal.copy()
        other_signal = other.signal.copy()

        if len(self_signal) > len(other_signal):
            diff = len(self_signal) - len(other_signal)
            other_signal.extend([0] * diff)
            new_duration = self.duration
        elif len(self_signal) < len(other_signal):
            diff = len(other_signal) - len(self_signal)
            self_signal.extend([0] * diff)
            new_duration = other.duration
        else:
            new_duration = max(self.duration, other.duration)

        return self_signal, other_signal, new_duration

    def __add__(self, other: "Signal"):
        if self.sample_rate != other.sample_rate:
            raise ValueError("Sample rate mismatch")

        s1, s2, new_duration = self.pad(other)
        new_signal = [a + b for a, b in zip(s1, s2)]
        new_amplitude = max(abs(x) for x in new_signal)

        return Signal(new_signal, new_amplitude, new_duration,
                      self.start_time, self.period, self.sample_rate)

    def __sub__(self, other: "Signal"):
        if self.sample_rate != other.sample_rate:
            raise ValueError("Sample rate mismatch")

        s1, s2, new_duration = self.pad(other)
        new_signal = [a - b for a, b in zip(s1, s2)]
        new_amplitude = max(abs(x) for x in new_signal)

        return Signal(new_signal, new_amplitude, new_duration,
                      self.start_time, self.period, self.sample_rate)

    def __mul__(self, other: "Signal"):
        if self.sample_rate != other.sample_rate:
            raise ValueError("Sample rate mismatch")

        s1, s2, new_duration = self.pad(other)
        new_signal = [a * b for a, b in zip(s1, s2)]
        new_amplitude = max(abs(x) for x in new_signal)

        return Signal(new_signal, new_amplitude, new_duration,
                      self.start_time, self.period, self.sample_rate)

    def __truediv__(self, other: "Signal"):
        if self.sample_rate != other.sample_rate:
            raise ValueError("Sample rate mismatch")

        s1, s2, new_duration = self.pad(other)
        new_signal = [
            a / b if b != 0 else 0
            for a, b in zip(s1, s2)
        ]
        new_amplitude = max(abs(x) for x in new_signal)

        return Signal(new_signal, new_amplitude, new_duration,
                      self.start_time, self.period, self.sample_rate)

    def __str__(self):
        header = f"{self.amplitude};{self.duration};{self.start_time};{self.period};{self.sample_rate}\n"
        values = " ".join(str(x) for x in self.signal)
        return header + values + "\n"

    def from_string(self, string: str):
        lines = string.split("\n")
        stats = lines[0].split(";")

        self.amplitude = float(stats[0])
        self.duration = float(stats[1])
        self.start_time = float(stats[2])
        self.period = float(stats[3])
        self.sample_rate = int(stats[4])

        self.signal = [float(x) for x in lines[1].split()]
        self.time = [
            self.start_time + i / self.sample_rate
            for i in range(len(self.signal))
        ]

    def sample(self, sample_rate: int):
        """
        Sample signal by returning a signal with a smaller sample rate.
        Returns new Signal object.
        """
        if sample_rate > self.sample_rate:
            raise ValueError("New sample rate cannot be higher than the original sample rate")
        if self.sample_rate % sample_rate != 0:
            raise ValueError("New sample rate must be an integer divisor of the original sample rate")

        step = self.sample_rate // sample_rate
        new_signal = self.signal[::step]  # slice with a given stride of step

        return Signal(new_signal, self.amplitude, self.duration,
                      self.start_time, self.period, sample_rate, True)

    def quantize_trunc(self, levels: int):
        """
        Quantize signal using truncation uniform quantization.
        'levels' should be an odd number to keep symmetry around zero.
        Returns new Signal object.
        """
        if levels < 1:
            raise ValueError("Number of levels must be at least 1")

        step = (2 * self.amplitude) / levels

        new_signal = []
        for x in self.signal:
            # math.trunc for truncating towards zero
            level = math.trunc(x / step)
            max_level = levels // 2
            level = max(-max_level, min(max_level - 1, level))
            new_signal.append(level * step)

        return Signal(new_signal, self.amplitude, self.duration,
                      self.start_time, self.period, self.sample_rate)

    def quantize_mid_rise(self, levels: int):
        """
        Quantize signal using mid-rise uniform quantization.
        'levels' should be an even number to keep symmetry around zero.
        Returns new Signal object.
        """
        if levels < 1:
            raise ValueError("Number of levels must be at least 1")

        step = (2 * self.amplitude) / levels

        new_signal = []
        for x in self.signal:
            # Floor into step index, then shift to center of that step
            level = math.floor(x / step)
            # Clamp to valid range
            max_level = levels // 2
            level = max(-max_level, min(max_level - 1, level))
            new_signal.append((level + 0.5) * step)

        return Signal(new_signal, self.amplitude, self.duration,
                      self.start_time, self.period, self.sample_rate)

    def convolve(self, other: "Signal"):
        if self.sample_rate != other.sample_rate:
            raise ValueError("Sample rate mismatch")

        n = len(self.signal)
        m = len(other.signal)
        output_len = n + m - 1
        result = [0.0] * output_len

        for i in range(n):
            for j in range(m):
                result[i + j] += self.signal[i] * other.signal[j]

        new_duration = self.duration + other.duration
        new_amplitude = max(abs(x) for x in result)

        return Signal(result, new_amplitude, new_duration,
                      self.start_time, self.period, self.sample_rate)

    def reconstruct_zoh(self, target_sample_rate: int):
        """
        Zero-order hold reconstruction for the signal. Direct interpolation method.
        Returns new Signal object.
        """
        factor = target_sample_rate // self.sample_rate
        upsampled = []
        for s in self.signal:
            upsampled.extend([s] * factor)  # repeat each sample 'factor' times
        return Signal(upsampled, self.amplitude, self.duration,
                      self.start_time, self.period, target_sample_rate)

    def reconstruct_foh(self, target_sample_rate: int):
        """
        First-order hold reconstruction for the signal. Direct interpolation method.
        Returns new Signal object.
        """
        factor = target_sample_rate // self.sample_rate
        upsampled = []

        for i in range(len(self.signal) - 1):
            s0 = self.signal[i]
            s1 = self.signal[i + 1]
            for j in range(factor):
                upsampled.append(s0 + (s1 - s0) * j / factor)

        upsampled.append(self.signal[-1])

        new_duration = (len(self.signal) - 1) / self.sample_rate

        return Signal(upsampled, self.amplitude, new_duration,
                      self.start_time, self.period, target_sample_rate)

    def reconstruct_sinc(self, target_sample_rate: int, kernel_size: int = 512):
        if kernel_size < 0:
            raise ValueError("Kernel size must be bigger than 0!")

        factor = target_sample_rate // self.sample_rate

        # Build sinc kernel centered at zero, windowed to kernel_size samples
        half = kernel_size // 2
        kernel_signal = []
        for i in range(-half, half):
            t = i / factor
            kernel_signal.append(np.sinc(t))  # math.sinc is normalized

        kernel = Signal(kernel_signal, max(abs(x) for x in kernel_signal),
                        len(kernel_signal) / target_sample_rate,
                        0.0, 0.0, target_sample_rate)

        # Upsample: insert (factor-1) zeros between each sample
        upsampled = []
        for s in self.signal:
            upsampled.append(s)
            upsampled.extend([0.0] * (factor - 1))

        upsampled_sig = Signal(upsampled, self.amplitude, self.duration,
                               self.start_time, self.period, target_sample_rate)

        convolved = upsampled_sig.convolve(kernel)

        # trim the leading 'half' samples to realign the output with the original signal
        # crop to the expected output length
        expected_len = (len(self.signal) - 1) * factor + 1
        trimmed = convolved.signal[half: half + expected_len]
        new_amplitude = max(abs(x) for x in trimmed) if trimmed else 0.0

        new_duration = (len(self.signal) - 1) / self.sample_rate

        return Signal(trimmed, new_amplitude, new_duration,
                      self.start_time, self.period, target_sample_rate)


# ------------------
# Signal Generation
# ------------------

def generate_continuous_signal(amplitude, duration, start_time,
                               period, type: "SignalType",
                               coefficient=0.0, sample_rate=100):
    sample_amount = int(duration * sample_rate)
    signal = []

    for i in range(sample_amount):
        t = start_time + i / sample_rate

        match type:
            case SignalType.UNIFORM_NOISE:
                signal.append(random.uniform(-amplitude, amplitude))

            case SignalType.GAUSSIAN_NOISE:
                signal.append(random.gauss(0, amplitude))

            case SignalType.SINE:
                signal.append(amplitude * math.sin(2 * math.pi * (t / period)))

            case SignalType.HALF_WAVE_RECT_SINE:
                s = amplitude * math.sin(2 * math.pi * (t / period))
                signal.append(max(0, s))

            case SignalType.FULL_WAVE_RECT_SINE:
                s = amplitude * math.sin(2 * math.pi * (t / period))
                signal.append(abs(s))

            case SignalType.RECT:
                local_t = (t - start_time) % period
                duty = coefficient * period
                signal.append(amplitude if local_t < duty else 0)

            case SignalType.RECT_SYMMETRIC:
                local_t = (t - start_time) % period
                duty = coefficient * period
                signal.append(amplitude if local_t < duty else -amplitude)

            case SignalType.TRIANGULAR:
                local_t = (t - start_time) % period
                if local_t < coefficient * period:
                    signal.append((amplitude / (coefficient * period)) * local_t)
                else:
                    signal.append(
                        amplitude - (amplitude / ((1 - coefficient) * period)) *
                        (local_t - coefficient * period)
                    )

            case SignalType.HEAVISIDE_STEP:
                if t < coefficient:
                    signal.append(0)
                elif t == coefficient:
                    signal.append(amplitude / 2)
                else:
                    signal.append(amplitude)

    return Signal(signal, amplitude, duration, start_time, period, sample_rate)


def generate_discrete_signal(amplitude, duration, start_time,
                             period, type: "SignalType",
                             coefficient, sample_rate=100):
    sample_amount = int(duration * sample_rate)
    signal = []

    for i in range(sample_amount):
        t = start_time + i / sample_rate

        match type:
            case SignalType.DIRAC_DELTA:
                signal.append(amplitude if abs(t - coefficient) < 1 / sample_rate else 0)

            case SignalType.IMPULSE_NOISE:
                signal.append(amplitude if random.random() < coefficient else 0)

    return Signal(signal, amplitude, duration, start_time, period, sample_rate)


def _window(n: int, M: int, wtype: "WindowType") -> float:
    """Return w(n) for a given window of length M (n = 0..M-1)."""
    match wtype:
        case WindowType.RECTANGULAR:
            return 1.0
        case WindowType.HAMMING:  # eq. (5)
            return 0.53836 - 0.46164 * math.cos(2 * math.pi * n / M)
        case WindowType.HANNING:  # eq. (6)
            return 0.5 - 0.5 * math.cos(2 * math.pi * n / M)
        case WindowType.BLACKMAN:  # eq. (7)
            return (0.42
                    - 0.5 * math.cos(2 * math.pi * n / M)
                    + 0.08 * math.cos(4 * math.pi * n / M))


# --------
# Filters
# --------

def design_lowpass_fir(M: int, K: int,
                       window: "WindowType" = "rectangular") -> list[float]:
    """
    Design a lowpass FIR filter using the window method.

    Parameters
    M : int   – number of coefficients (should be odd)
    K : int   – frequency divider; cutoff frequency fo = fp / K
    window    – window function to apply

    Returns
    list[float] of length M  (h[0] … h[M-1])
    """
    if M % 2 == 0:
        raise ValueError("M should be odd for a symmetric FIR filter")
    centre = (M - 1) / 2
    h = []
    for n in range(M):
        shift = n - centre
        if shift == 0:
            h_n = 2.0 / K
        else:
            h_n = math.sin(2 * math.pi * shift / K) / (math.pi * shift)
        h_n *= _window(n, M, window)
        h.append(h_n)
    return h


def design_fir(M: int, K: int,
               window: "WindowType" = "rectangular",
               ftype: "FilterType" = "lowpass") -> list[float]:
    """
    Design a FIR filter of any supported type.

    Bandpass  (F1): multiply lowpass coefficients by s(n) = 2·sin(πn/2)
    Highpass  (F2): multiply lowpass coefficients by s(n) = (-1)^n
    """
    h = design_lowpass_fir(M, K, window)

    match ftype:
        case FilterType.LOWPASS:
            return h
        case FilterType.BANDPASS:  # eq. from sec. 7 of spec
            return [h[n] * 2 * math.sin(math.pi * n / 2) for n in range(M)]
        case FilterType.HIGHPASS:
            return [h[n] * ((-1) ** n) for n in range(M)]


def make_filter_signal(coefficients: list[float], sample_rate: int) -> Signal:
    """Wrap a coefficient list into a Signal object suitable for convolution."""
    amp = max(abs(x) for x in coefficients) if coefficients else 0.0
    dur = len(coefficients) / sample_rate
    return Signal(coefficients, amp, dur, 0.0, 0.0, sample_rate)


def filter_signal(x: Signal, M: int, K: int,
                  window: "WindowType" = "rectangular",
                  ftype: "FilterType" = "lowpass") -> Signal:
    """
    Filter signal x with an M-tap FIR filter designed for sample_rate=x.sample_rate.
    Returns the convolved (filtered) signal.
    """
    coeffs = design_fir(M, K, window, ftype)
    h_sig = make_filter_signal(coeffs, x.sample_rate)
    return x.convolve(h_sig)


# ------------------
# Cross-correlation
# ------------------

def cross_correlate_direct(h: Signal, x: Signal) -> Signal:
    """
    Cross-correlation via the direct formula.

    The output vector is re-indexed so that index 0 of the returned
    Signal corresponds to R_hx(-(N-1))  (leftmost output sample).
    Output length = M + N - 1.
    """
    if h.sample_rate != x.sample_rate:
        raise ValueError("Sample rate mismatch")

    M = len(h.signal)
    N = len(x.signal)
    out_len = M + N - 1
    result = [0.0] * out_len

    # n ranges over -(N-1) … M-1  →  stored at index n+(N-1)
    for idx in range(out_len):
        n = idx - (N - 1)
        val = 0.0
        for k in range(M):
            x_idx = k - n  # index into x (possibly negative / out of range)
            # x is zero outside [0, N-1]  (zero padding convention)
            # BUT note: correlation slides x FORWARD (not reversed like convolution)
            # R_hx(n) = Σ h(k)·x(k-n)  — re-check with eq. (8)/(9):
            # eq.(9):  Σ_{k=0}^{M-1} h(k)·x(n-k)
            # where x is zero outside [0,N-1]
            if 0 <= x_idx < N:
                val += h.signal[k] * x.signal[x_idx]
        result[idx] = val

    amp = max(abs(v) for v in result) if result else 0.0
    dur = (M + N - 1) / h.sample_rate
    # start_time shifted to represent negative-lag region
    start = -(N - 1) / h.sample_rate
    return Signal(result, amp, dur, start, 0.0, h.sample_rate)


def cross_correlate_via_convolution(h: Signal, x: Signal) -> Signal:
    """
    Cross-correlation via convolution.

    Equivalently (using the identity for finite sequences):
        R_hx = conv(h_rev, x)   where h_rev(k) = h(M-1-k)
    """
    if h.sample_rate != x.sample_rate:
        raise ValueError("Sample rate mismatch")

    # Reverse x
    x_rev = Signal(list(reversed(x.signal)),
                   x.amplitude, x.duration, x.start_time, x.period, x.sample_rate)
    conv_result = h.convolve(x_rev)

    # Adjust start_time to match direct implementation (lag axis)
    N = len(x.signal)
    conv_result.start_time = -(N - 1) / h.sample_rate
    conv_result.time = [conv_result.start_time + i / conv_result.sample_rate
                        for i in range(len(conv_result.signal))]
    return conv_result


# ----------------
# Complex Signals
# ----------------

class ComplexSignal:
    """
    Represents a discrete signal with complex-valued samples.

    Used to store the result of a Fourier-type transform (F1/F2), i.e. a
    spectrum X(m), m = 0..N-1, corresponding to the frequency m*f0
    (eq. F-3: f0 = fpr / N).
    """

    def __init__(self, signal=None, sample_rate=100):
        self.signal = [complex(v) for v in signal] if signal is not None else []
        self.sample_rate = sample_rate
        self.N = len(self.signal)
        self.f0 = self.sample_rate / self.N if self.N > 0 else 0.0
        self.freq = [m * self.f0 for m in range(self.N)]

    def __str__(self):
        """Serialize to text: header line 'sample_rate;N' + 're,im' pairs."""
        header = f"{self.sample_rate};{self.N}\n"
        values = " ".join(f"{v.real},{v.imag}" for v in self.signal)
        return header + values + "\n"

    def from_string(self, string: str):
        """Read a complex signal previously written with __str__."""
        lines = string.strip().split("\n")
        stats = lines[0].split(";")
        self.sample_rate = int(stats[0])

        self.signal = []
        if len(lines) > 1 and lines[1].strip():
            for token in lines[1].split():
                re_str, im_str = token.split(",")
                self.signal.append(complex(float(re_str), float(im_str)))

        self.N = len(self.signal)
        self.f0 = self.sample_rate / self.N if self.N > 0 else 0.0
        self.freq = [m * self.f0 for m in range(self.N)]

    def real_part(self):
        return [v.real for v in self.signal]

    def imag_part(self):
        return [v.imag for v in self.signal]

    def magnitude(self):
        return [abs(v) for v in self.signal]

    def phase(self):
        return [cmath.phase(v) for v in self.signal]

    def plot_w1(self, title="Widmo sygnalu (W1)"):
        """
        (W1) - gorny wykres: czesc rzeczywista amplitudy w funkcji
        czestotliwosci, dolny wykres: czesc urojona.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
        fig.suptitle(title)

        ax1.stem(self.freq, self.real_part())
        ax1.set_ylabel("Re{X(m)}")
        ax1.grid(True)

        ax2.stem(self.freq, self.imag_part())
        ax2.set_ylabel("Im{X(m)}")
        ax2.set_xlabel("f [Hz]")
        ax2.grid(True)

        fig.tight_layout()
        return fig

    def plot_w2(self, title="Widmo sygnalu (W2)"):
        """
        (W2) - gorny wykres: modul liczby zespolonej w funkcji
        czestotliwosci, dolny wykres: argument liczby zespolonej.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
        fig.suptitle(title)

        ax1.stem(self.freq, self.magnitude())
        ax1.set_ylabel("|X(m)|")
        ax1.grid(True)

        ax2.stem(self.freq, self.phase())
        ax2.set_ylabel("arg{X(m)} [rad]")
        ax2.set_xlabel("f [Hz]")
        ax2.grid(True)

        fig.tight_layout()
        return fig


# -----------
# Transforms
# -----------
def bit_reverse(x, N, n):
    """
    Auxiliary function that permutes a list according to the
    bit-reversed order of indices (used as the first step of the
    iterative, in-place radix-2 FFT).

    x - list to permute (modified in place and returned)
    N - length of x
    n - number of bits, i.e. n = log2(N)
    """
    for i in range(N):
        j = 0
        for k in range(0, n):
            j <<= 1
            j += (i >> k) & 1
        if j > i:
            x[i], x[j] = x[j], x[i]
    return x


def _check_power_of_two(N: int):
    """Verify N = 2**n for n in [1, 10], as required by the task description."""
    if N == 0 or (N & (N - 1)) != 0:
        raise ValueError("Signal length must be a power of two (2^n, n=1..10)")


def dft(signal: "Signal") -> "ComplexSignal":
    """
    (F1) Discrete Fourier Transform, computed directly from the
    definition (F-1)/(F-5). Complexity O(N^2).
    """
    x = signal.signal
    N = len(x)
    _check_power_of_two(N)

    X = []
    for m in range(N):
        acc = 0j
        for n in range(N):
            acc += x[n] * cmath.exp(-2j * math.pi * m * n / N)
        X.append(acc / N)

    return ComplexSignal(X, signal.sample_rate)


def idft(spectrum: "ComplexSignal") -> "Signal":
    """
    Inverse Discrete Fourier Transform, computed directly from the
    definition (F-2). Complexity O(N^2). Returns the real part of
    x(n) as a Signal (the imaginary part is dropped, as expected for
    a real-valued input signal).
    """
    X = spectrum.signal
    N = len(X)
    _check_power_of_two(N)

    x = []
    for n in range(N):
        acc = 0j
        for m in range(N):
            acc += X[m] * cmath.exp(2j * math.pi * m * n / N)
        x.append(acc.real)

    amp = max((abs(v) for v in x), default=0.0)
    return Signal(x, amp, N / spectrum.sample_rate, 0.0, 0.0, spectrum.sample_rate)


def _fft_iterative(x: list, inverse: bool) -> list:
    """
    Core iterative radix-2 FFT (decimation in time, in-place).

    The input vector is first reordered using `bit_reverse`, so that
    it can be treated as N one-point transforms, then combined
    bottom-up (2-point, 4-point, ... N-point) using the butterfly
    structure described by eq. (F-12).

    inverse=False -> forward transform with kernel exp(-j*2*pi*k/size),
                      result divided by N (eq. F-1/F-5).
    inverse=True  -> inverse transform with kernel exp(+j*2*pi*k/size),
                      result NOT divided by N (eq. F-2).
    """
    N = len(x)
    _check_power_of_two(N)

    n_bits = N.bit_length() - 1  # log2(N)
    x = bit_reverse(list(x), N, n_bits)

    sign = 1 if inverse else -1
    size = 2
    while size <= N:
        half = size // 2
        w_step = cmath.exp(sign * 2j * math.pi / size)  # W_size^{sign}
        for start in range(0, N, size):
            w = 1 + 0j
            for k in range(half):
                t = w * x[start + k + half]
                u = x[start + k]
                x[start + k] = u + t  # F-12, first row
                x[start + k + half] = u - t  # F-12, second row
                w *= w_step
        size *= 2

    if not inverse:
        x = [v / N for v in x]

    return x


def dit_fft(signal: "Signal") -> "ComplexSignal":
    """
    (F1) Fast Fourier Transform with decimation in time (DIT FFT),
    implemented iteratively and in-place, based on `bit_reverse`.
    Accepts signals whose length is a power of two (2^n, n=1..10).
    Complexity O(N log2 N).
    """
    x = [complex(v) for v in signal.signal]
    X = _fft_iterative(x, inverse=False)
    return ComplexSignal(X, signal.sample_rate)


def idit_fft(spectrum: "ComplexSignal") -> "Signal":
    """
    Inverse of `dit_fft`, iterative, based on `bit_reverse`.
    Per the task description, the inverse transform uses a kernel
    without the minus sign in the exponent and the result is NOT
    divided by N. Returns the real part of x(n) as a Signal.
    """
    X = [complex(v) for v in spectrum.signal]
    x = _fft_iterative(X, inverse=True)
    real_part = [v.real for v in x]

    amp = max((abs(v) for v in real_part), default=0.0)
    return Signal(real_part, amp, len(real_part) / spectrum.sample_rate,
                  0.0, 0.0, spectrum.sample_rate)


def compare_dft_fft_speed(signal: "Signal") -> dict:
    """
    Helper for the report (F1): compares the execution time of the
    direct DFT (algorithm from the definition) and the iterative
    DIT FFT for the given signal.

    Returns a dict {"dft": (ComplexSignal, time_seconds),
                     "fft": (ComplexSignal, time_seconds)}.
    """
    import time

    t0 = time.perf_counter()
    X_dft = dft(signal)
    t_dft = time.perf_counter() - t0

    t0 = time.perf_counter()
    X_fft = dit_fft(signal)
    t_fft = time.perf_counter() - t0

    return {
        "dft": (X_dft, t_dft),
        "fft": (X_fft, t_fft),
    }

# -------------------------
# FFT DIF (Decimation in Frequency)
# -------------------------

def dif_fft(signal: Signal) -> ComplexSignal:
    """
    FFT radix-2 DIF (Decimation In Frequency).
    """
    x = [complex(v) for v in signal.signal]

    N = len(x)
    _check_power_of_two(N)

    size = N
    while size >= 2:
        half = size // 2

        for start in range(0, N, size):
            for k in range(half):
                a = x[start + k]
                b = x[start + k + half]

                x[start + k] = a + b

                twiddle = cmath.exp(-2j * math.pi * k / size)
                x[start + k + half] = (a - b) * twiddle

        size //= 2

    n_bits = N.bit_length() - 1
    x = bit_reverse(x, N, n_bits)

    x = [v / N for v in x]

    return ComplexSignal(x, signal.sample_rate)


def idif_fft(spectrum: ComplexSignal) -> Signal:
    """
    Reverse FFT DIF.
    """
    x = [complex(v) for v in spectrum.signal]

    N = len(x)
    _check_power_of_two(N)

    n_bits = N.bit_length() - 1
    x = bit_reverse(x, N, n_bits)

    size = 2

    while size <= N:
        half = size // 2

        for start in range(0, N, size):
            for k in range(half):
                twiddle = cmath.exp(2j * math.pi * k / size)

                a = x[start + k]
                b = x[start + k + half]

                x[start + k] = a + b * twiddle
                x[start + k + half] = a - b * twiddle

        size *= 2

    real = [v.real for v in x]

    return Signal(
        real,
        max(abs(v) for v in real),
        N / spectrum.sample_rate,
        0.0,
        0.0,
        spectrum.sample_rate
    )

# -----------------
# Wavelet Transform
# -----------------

def _normalize_filter(h: list) -> list:
    """Scale filter coefficients so that sum(h_k^2) == 1 (eq. TF-5)."""
    norm = math.sqrt(sum(c * c for c in h))
    return [c / norm for c in h]


def _qmf_highpass(h: list) -> list:
    """
    Derive the high-pass (detail) filter G from the low-pass (scaling)
    filter H using the quadrature mirror relation
    g_k = (-1)^k * h_{K-1-k} (eq. TF-9).
    """
    K = len(h)
    return [((-1) ** k) * h[K - 1 - k] for k in range(K)]


def _db4_lowpass() -> list:
    """Daubechies 4 (K=4) low-pass coefficients (eq. TF-9)."""
    sqrt3 = math.sqrt(3.0)
    sqrt2 = math.sqrt(2.0)
    h0 = (1 + sqrt3) / (4 * sqrt2)
    h1 = (3 + sqrt3) / (4 * sqrt2)
    h2 = (3 - sqrt3) / (4 * sqrt2)
    h3 = (1 - sqrt3) / (4 * sqrt2)
    return _normalize_filter([h0, h1, h2, h3])


# Low-pass (scaling) filters H for the supported wavelets.
# DB6 and DB8 coefficients are given in the handout (eq. TF-10, TF-11)
# scaled so that sum(h_k) = 2; they are re-normalised here so that
# sum(h_k^2) = 1, as required by the orthogonality conditions (eq. TF-5).
WAVELET_FILTERS = {
    "db4": _db4_lowpass(),
    "db6": _normalize_filter([0.47046721, 1.14111692, 0.650365,
                              -0.19093442, -0.12083221, 0.0498175]),
    "db8": _normalize_filter([0.32580343, 1.01094572, 0.8922014, -0.03957503,
                              -0.26450717, 0.0436163, 0.0465036, -0.01498699]),
}


def dwt_one_level(signal: "Signal", wavelet: str = "db4"):
    """
    (T3) Single-level discrete wavelet transform.

    The N-point input signal x(n) is filtered with the low-pass filter H
    and the high-pass filter G (eq. TF-1/TF-2, circular convolution so
    that periodic boundary conditions are used), and each branch is
    down-sampled by 2 (block "v2"), producing two N/2-point signals:

      - x1(n): approximation (low-pass / scaling) coefficients
      - x2(n): detail (high-pass / wavelet) coefficients

    wavelet: one of "db4", "db6", "db8"

    Returns (x1, x2) as Signal objects (each of length N/2).
    """
    if wavelet not in WAVELET_FILTERS:
        raise ValueError(f"Unsupported wavelet '{wavelet}', "
                         f"available: {list(WAVELET_FILTERS)}")

    x = signal.signal
    N = len(x)
    _check_power_of_two(N)
    if N < 2:
        raise ValueError("Signal must contain at least 2 samples")

    h = WAVELET_FILTERS[wavelet]
    g = _qmf_highpass(h)
    K = len(h)
    half = N // 2

    x1 = [0.0] * half
    x2 = [0.0] * half
    for n in range(half):
        sh = 0.0
        sg = 0.0
        for k in range(K):
            sample = x[(2 * n - k) % N]
            sh += h[k] * sample
            sg += g[k] * sample
        x1[n] = sh
        x2[n] = sg

    new_sr = max(1, signal.sample_rate // 2)
    sig1 = Signal(x1, max((abs(v) for v in x1), default=0.0),
                  signal.duration / 2, signal.start_time, signal.period, new_sr)
    sig2 = Signal(x2, max((abs(v) for v in x2), default=0.0),
                  signal.duration / 2, signal.start_time, signal.period, new_sr)
    return sig1, sig2


def idwt_one_level(approx: "Signal", detail: "Signal", wavelet: str = "db4") -> "Signal":
    """
    Inverse of `dwt_one_level`. Reconstructs the N-point signal x(n)
    from its approximation x1(n) and detail x2(n) coefficients (each of
    length N/2), using the same filters H and G as the forward
    transform (the synthesis operation is the adjoint of the analysis
    operation, which is exact for the orthogonal Daubechies filters
    used here, eq. TF-3).
    """
    if wavelet not in WAVELET_FILTERS:
        raise ValueError(f"Unsupported wavelet '{wavelet}', "
                         f"available: {list(WAVELET_FILTERS)}")
    if len(approx.signal) != len(detail.signal):
        raise ValueError("Approximation and detail signals must have equal length")

    h = WAVELET_FILTERS[wavelet]
    g = _qmf_highpass(h)
    K = len(h)

    half = len(approx.signal)
    N = half * 2

    # "v2": insert zeros at the positions dropped during analysis
    up1 = [0.0] * N
    up2 = [0.0] * N
    for n in range(half):
        up1[2 * n] = approx.signal[n]
        up2[2 * n] = detail.signal[n]

    out = [0.0] * N
    for n in range(N):
        s = 0.0
        for k in range(K):
            idx = (n + k) % N
            s += h[k] * up1[idx] + g[k] * up2[idx]
        out[n] = s

    new_sr = approx.sample_rate * 2
    amp = max((abs(v) for v in out), default=0.0)
    return Signal(out, amp, approx.duration * 2, approx.start_time, approx.period, new_sr)

# -------------------------
# DCT-II
# -------------------------

def dct2(signal: Signal) -> Signal:
    """
    DCT-II function.
    """

    x = signal.signal
    N = len(x)

    _check_power_of_two(N)

    X = []

    for m in range(N):

        if m == 0:
            c = math.sqrt(1 / N)
        else:
            c = math.sqrt(2 / N)

        acc = 0.0

        for n in range(N):
            acc += x[n] * math.cos(
                math.pi * (2 * n + 1) * m / (2 * N)
            )

        X.append(c * acc)

    return Signal(
        X,
        max(abs(v) for v in X),
        signal.duration,
        signal.start_time,
        signal.period,
        signal.sample_rate
    )


def idct2(signal: Signal) -> Signal:
    """
    Inverse DCT-II.
    """

    X = signal.signal
    N = len(X)

    x = []

    for n in range(N):

        acc = 0.0

        for m in range(N):

            if m == 0:
                c = math.sqrt(1 / N)
            else:
                c = math.sqrt(2 / N)

            acc += (
                c *
                X[m] *
                math.cos(
                    math.pi * (2 * n + 1) * m / (2 * N)
                )
            )

        x.append(acc)

    return Signal(
        x,
        max(abs(v) for v in x),
        signal.duration,
        signal.start_time,
        signal.period,
        signal.sample_rate
    )

# -------------------------
# Fast DCT-II (FCT-II)
# -------------------------

def fct2(signal: Signal) -> Signal:

    x = signal.signal
    N = len(x)

    _check_power_of_two(N)

    y = [0.0] * N

    half = N // 2

    for n in range(half):
        y[n] = x[2 * n]
        y[N - 1 - n] = x[2 * n + 1]

    Y = dit_fft(
        Signal(
            y,
            signal.amplitude,
            signal.duration,
            signal.start_time,
            signal.period,
            signal.sample_rate
        )
    )

    X = []

    for m in range(N):

        if m == 0:
            c = math.sqrt(1 / N)
        else:
            c = math.sqrt(2 / N)

        factor = cmath.exp(
            -1j * math.pi * m / (2 * N)
        )

        value = (c * factor * Y.signal[m]).real

        X.append(value)

    return Signal(
        X,
        max(abs(v) for v in X),
        signal.duration,
        signal.start_time,
        signal.period,
        signal.sample_rate
    )

# -------------------------
# Walsh-Hadamard
# -------------------------

def hadamard_matrix(order: int):

    if order == 0:
        return np.array([[1.0]])

    H = hadamard_matrix(order - 1)

    return (
        1 / math.sqrt(2)
    ) * np.block([
        [H, H],
        [H, -H]
    ])


def walsh_hadamard(signal: Signal) -> Signal:
    """
    Walsh-Hadamard transform.
    """

    N = len(signal.signal)

    _check_power_of_two(N)

    m = int(math.log2(N))

    H = hadamard_matrix(m)

    X = H @ np.array(signal.signal)

    X = X.tolist()

    return Signal(
        X,
        max(abs(v) for v in X),
        signal.duration,
        signal.start_time,
        signal.period,
        signal.sample_rate
    )

# -------------------------
# Fast Walsh-Hadamard
# -------------------------

def fast_walsh_hadamard(signal: Signal) -> Signal:
    """
    Fast Walsh-Hadamard transform.
    O(N log2 N)
    """

    x = list(signal.signal)

    N = len(x)

    _check_power_of_two(N)

    h = 1

    while h < N:

        for i in range(0, N, h * 2):

            for j in range(i, i + h):

                a = x[j]
                b = x[j + h]

                x[j] = a + b
                x[j + h] = a - b

        h *= 2

    scale = 1 / math.sqrt(N)

    x = [v * scale for v in x]

    return Signal(
        x,
        max(abs(v) for v in x),
        signal.duration,
        signal.start_time,
        signal.period,
        signal.sample_rate
    )


def ifast_walsh_hadamard(signal: Signal) -> Signal:
    """
    WHT is ortogonal:
    H^-1 = H^T = H
    """

    return fast_walsh_hadamard(signal)

# --------
# Metrics
# --------

def mse(original: Signal, quantized: Signal):
    """Mean Squared Error"""
    s1, s2, _ = original.pad(quantized)
    n = len(s1)
    return sum((a - b) ** 2 for a, b in zip(s1, s2)) / n


def snr(original: Signal, quantized: Signal):
    """Signal-to-Noise Ratio (dB)"""
    s1, s2, _ = original.pad(quantized)
    signal_power = sum(a ** 2 for a in s1)
    noise_power = sum((a - b) ** 2 for a, b in zip(s1, s2))
    if noise_power == 0:
        return float('inf')
    return 10 * math.log10(signal_power / noise_power)


def psnr(original: Signal, quantized: Signal):
    """Peak Signal-to-Noise Ratio (dB)"""
    error = mse(original, quantized)
    if error == 0:
        return float('inf')
    peak = max(abs(x) for x in original.signal)
    return 10 * math.log10(peak ** 2 / error)


def md(original: Signal, quantized: Signal):
    """Maximum Difference"""
    s1, s2, _ = original.pad(quantized)
    return max(abs(a - b) for a, b in zip(s1, s2))


# ------
# Enums
# ------

class SignalType(Enum):
    UNIFORM_NOISE = 0
    GAUSSIAN_NOISE = 1
    SINE = 2
    HALF_WAVE_RECT_SINE = 3
    FULL_WAVE_RECT_SINE = 4
    RECT = 5
    RECT_SYMMETRIC = 6
    TRIANGULAR = 7
    HEAVISIDE_STEP = 8
    DIRAC_DELTA = 9
    IMPULSE_NOISE = 10


class WindowType(Enum):
    RECTANGULAR = "rectangular"
    HAMMING = "hamming"  # O1
    HANNING = "hanning"  # O2
    BLACKMAN = "blackman"  # O3


class FilterType(Enum):
    LOWPASS = "lowpass"
    BANDPASS = "bandpass"  # F1 – środkowoprzepustowy
    HIGHPASS = "highpass"  # F2 – górnoprzepustowy