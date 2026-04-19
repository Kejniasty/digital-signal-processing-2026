import math
import random
from enum import Enum


class Signal:
    """Represents a continuous or discrete signal."""

    def __init__(self, signal=None, amplitude=0.0, duration=0.0,
                 start_time=0.0, period=0.0, sample_rate=100):
        self.signal = signal.copy() if signal is not None else []
        self.amplitude = amplitude
        self.duration = duration
        self.start_time = start_time
        self.period = period
        self.sample_rate = sample_rate

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
        
    def discretize(self):
        return



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
