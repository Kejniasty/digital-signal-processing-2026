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

        return Signal(trimmed, new_amplitude, self.duration,
                      self.start_time, self.period, target_sample_rate)



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
