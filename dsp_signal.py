import math
import random
import numpy as np
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
        new_signal = self.signal[::step] #slice with a given stride of step

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


#------------------
# Signal Generation
#------------------

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
        case WindowType.HAMMING:            # eq. (5)
            return 0.53836 - 0.46164 * math.cos(2 * math.pi * n / M)
        case WindowType.HANNING:            # eq. (6)
            return 0.5 - 0.5 * math.cos(2 * math.pi * n / M)
        case WindowType.BLACKMAN:           # eq. (7)
            return (0.42
                    - 0.5  * math.cos(2 * math.pi * n / M)
                    + 0.08 * math.cos(4 * math.pi * n / M))

#--------
# Filters
#--------

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

#------------------
# Cross-correlation
#------------------

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

#--------
# Metrics
#--------

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

#------
# Enums
#------

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
