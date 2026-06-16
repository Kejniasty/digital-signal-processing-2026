import struct
from dsp_signal import Signal, ComplexSignal

SIGNAL_REAL = 0
SIGNAL_COMPLEX = 1


def signal_to_file(signal, filename: str, signal_type_name: str):
    """Save Signal or ComplexSignal to a binary file."""

    with open(filename, "wb") as f:

        # --------------------------------------------------
        # signal kind
        # --------------------------------------------------
        if isinstance(signal, ComplexSignal):
            signal_kind = SIGNAL_COMPLEX
        else:
            signal_kind = SIGNAL_REAL

        f.write(struct.pack("B", signal_kind))

        # --------------------------------------------------
        # metadata
        # --------------------------------------------------

        if signal_kind == SIGNAL_REAL:

            num_samples = len(signal.signal)

            f.write(struct.pack("d", signal.start_time))
            f.write(struct.pack("d", signal.sample_rate))
            f.write(struct.pack("d", signal.duration))
            f.write(struct.pack("d", signal.amplitude))
            f.write(struct.pack("d", signal.period))
            f.write(struct.pack("d", signal.is_sampled))

            encoded = signal_type_name.encode("utf-8")
            f.write(struct.pack("I", len(encoded)))
            f.write(encoded)

            f.write(struct.pack("I", num_samples))

            for value in signal.signal:
                f.write(struct.pack("d", value))

        else:

            num_samples = len(signal.signal)

            f.write(struct.pack("d", signal.sample_rate))

            encoded = signal_type_name.encode("utf-8")
            f.write(struct.pack("I", len(encoded)))
            f.write(encoded)

            f.write(struct.pack("I", num_samples))

            for value in signal.signal:
                f.write(struct.pack("d", value.real))
                f.write(struct.pack("d", value.imag))


def signal_from_file(filename: str):
    """Load Signal or ComplexSignal from a binary file."""

    with open(filename, "rb") as f:

        signal_kind = struct.unpack("B", f.read(1))[0]

        # --------------------------------------------------
        # REAL SIGNAL
        # --------------------------------------------------

        if signal_kind == SIGNAL_REAL:

            start_time = struct.unpack("d", f.read(8))[0]
            sample_rate = struct.unpack("d", f.read(8))[0]
            duration = struct.unpack("d", f.read(8))[0]
            amplitude = struct.unpack("d", f.read(8))[0]
            period = struct.unpack("d", f.read(8))[0]
            is_sampled = struct.unpack("d", f.read(8))[0]

            name_len = struct.unpack("I", f.read(4))[0]
            type_name = f.read(name_len).decode("utf-8")

            num_samples = struct.unpack("I", f.read(4))[0]

            values = [
                struct.unpack("d", f.read(8))[0]
                for _ in range(num_samples)
            ]

            signal = Signal(
                signal=values,
                amplitude=amplitude,
                duration=duration,
                start_time=start_time,
                period=period,
                sample_rate=int(sample_rate),
                is_sampled=is_sampled,
            )

            return signal, type_name

        # --------------------------------------------------
        # COMPLEX SIGNAL
        # --------------------------------------------------

        elif signal_kind == SIGNAL_COMPLEX:

            sample_rate = struct.unpack("d", f.read(8))[0]

            name_len = struct.unpack("I", f.read(4))[0]
            type_name = f.read(name_len).decode("utf-8")

            num_samples = struct.unpack("I", f.read(4))[0]

            values = []

            for _ in range(num_samples):
                real = struct.unpack("d", f.read(8))[0]
                imag = struct.unpack("d", f.read(8))[0]

                values.append(complex(real, imag))

            signal = ComplexSignal(
                signal=values,
                sample_rate=int(sample_rate)
            )

            return signal, type_name

        else:
            raise ValueError("Unknown signal type in file")