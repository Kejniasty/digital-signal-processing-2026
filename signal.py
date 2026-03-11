import math
import random
from enum import Enum


class Signal(object):
    """Class to represent a signal, needs an amplitude, duration, start time and period."""

    def __init__(self, signal: list, amplitude: float, duration: float, start_time: float, period: float):
        """Base constructor for signal object."""
        self.signal = signal.copy()
        self.amplitude = amplitude
        self.start_time = start_time
        self.duration = duration
        self.period = period
        # As the signal is not continuous in any case because of coding limitations we need the info for comparing the two signals.
        # If their sample rates were not the same it would break most of the calculations
        # Probably should be calculated from the len of the signal and duration, for now it isn't changing anywhere so this is fine.
        self.sample_rate = SAMPLE_RATE

    def pad(self, other: "Signal"):
        """Pad the shorter signal with enough zeros - returns copies of both of the signals."""

        other_signal = other.signal.copy()
        self_signal = self.signal.copy()
        new_duration = 0

        if self.duration > other.duration:
            other_signal = pad_array(other_signal, len(self.signal) - len(other.signal))
            new_duration = self.duration
        elif self.duration < other.duration:
            self_signal = pad_array(self_signal, len(other.signal) - len(self.signal))
            new_duration = other.duration
        # if neither is true it means that they are equal in length and padding is not needed

        return self_signal, other_signal, new_duration


    def __add__(self, other: "Signal"):
        """Add two signals together - this also affects the amplitude and duration of the resulting signal."""
        if self.sample_rate != other.sample_rate:
            raise ValueError("Sample rate mismatch")

        new_signal = []
        self_signal, other_signal, new_duration = self.pad(other)

        for i in range(len(self_signal)):
            new_signal.append(other_signal[i] + self_signal[i])

        return Signal(new_signal, self.amplitude + other.amplitude, new_duration, self.start_time, self.period)

    def __sub__(self, other):
        """Subtract other from self - this also affects amplitude and duration of the resulting signal."""
        if self.sample_rate != other.sample_rate:
            raise ValueError("Sample rate mismatch")

        new_signal = []
        self_signal, other_signal, new_duration = self.pad(other)

        for i in range(len(self_signal)):
            new_signal.append(self_signal[i] - other_signal[i])

        return Signal(new_signal, self.amplitude - other.amplitude, new_duration, self.start_time, self.period)

    def __mul__(self, other: "Signal"):
        """Multiplies two signals together - this also affects the amplitude and duration of the resulting signal."""
        if self.sample_rate != other.sample_rate:
            raise ValueError("Sample rate mismatch")

        new_signal = []
        self_signal, other_signal, new_duration = self.pad(other)

        for i in range(len(self_signal)):
            new_signal.append(other_signal[i] * self_signal[i])

        return Signal(new_signal, self.amplitude * other.amplitude, new_duration, self.start_time, self.period)

    def __truediv__(self, other: "Signal"):
        """Divides self by other - this also affects the amplitude and duration of the resulting signal."""
        if self.sample_rate != other.sample_rate:
            raise ValueError("Sample rate mismatch")

        new_signal = []
        self_signal, other_signal, new_duration = self.pad(other)

        for i in range(len(self_signal)):
            new_signal.append(self_signal[i] / other_signal[i])

        return Signal(new_signal, self.amplitude / other.amplitude, new_duration, self.start_time, self.period)

    def __str__(self):
        output = f"{self.amplitude};{self.duration};{self.start_time};{self.period};\n"
        for signal in self.signal:
            output += f"{signal} "
        output += "\n"
        return output


def pad_array(array: list, length: int):
    """Pad the given array with length amount of zeros."""
    for i in range(length):
        array.append(0)
    return array


def generate_continuous_signal(amplitude: float, duration: float, start_time: float, period: float, type: "SignalType", fill_coefficent = 0.0):
    """Generates a pseudo-continous signal with a given amplitude, duration, start time, period and a type. Returns a Signal object."""
    signal = []
    sample_amount = int(round(duration * SAMPLE_RATE, 0))
    match type:
        case SignalType.UNIFORM_NOISE:
            for i in range(sample_amount):
                signal.append(random.uniform(-amplitude, amplitude))
        case SignalType.GAUSSIAN_NOISE:
            for i in range(sample_amount):
                signal.append(random.gauss() * amplitude)
        case SignalType.SINE:
            for i in range(sample_amount):
                signal.append(amplitude * math.sin((2 * math.pi / period) * (i - start_time)))
        case SignalType.HALF_WAVE_RECT_SINE:
            for i in range(sample_amount):
                signal.append(0.5 * amplitude * (math.sin((2 * math.pi / period) * (i - start_time))
                                                 + math.sin((2 * math.pi / period) * (i - start_time))))
        case SignalType.FULL_WAVE_RECT_SINE:
            for i in range(sample_amount):
                signal.append(amplitude * abs(math.sin((2 * math.pi / period) * (i - start_time))))
        case SignalType.RECT:
            for i in range(sample_amount):
                t = i / SAMPLE_RATE
                local_t = (t - start_time) % period

                if 0 <= local_t < fill_coefficent * period:
                    signal.append(amplitude)
                elif fill_coefficent * period <= local_t < period:
                    signal.append(0)
        case SignalType.RECT_SYMMETRIC:
            for i in range(sample_amount):
                t = i / SAMPLE_RATE
                local_t = (t - start_time) % period

                if 0 <= local_t < fill_coefficent * period:
                    signal.append(amplitude)
                elif fill_coefficent * period <= local_t < period:
                    signal.append(-amplitude)
        case SignalType.TRIANGULAR:
            for i in range(sample_amount):
                t = i / SAMPLE_RATE
                local_t = (t - start_time) % period

                if 0 <= local_t < fill_coefficent * period:
                    signal.append(amplitude)
                elif fill_coefficent * period <= local_t < period:
                    signal.append(-amplitude)

    return Signal(signal, amplitude, duration, start_time, period)


# typedef
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

SAMPLE_RATE = 100