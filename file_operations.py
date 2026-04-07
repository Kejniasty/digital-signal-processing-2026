import struct
from dsp_signal import Signal


def signal_to_file(signal: Signal, filename: str, signal_type_name: str):
    """Save a signal and its metadata to a binary file."""

    num_samples = len(signal.signal)

    with open(filename, "wb") as f:
        # Metadata
        f.write(struct.pack("d", signal.start_time))
        f.write(struct.pack("d", signal.sample_rate))
        f.write(struct.pack("d", signal.duration))
        f.write(struct.pack("d", signal.amplitude))
        f.write(struct.pack("d", signal.period))

        # Signal type as UTF-8 string (length + bytes)
        encoded = signal_type_name.encode("utf-8")
        f.write(struct.pack("I", len(encoded)))
        f.write(encoded)

        # Number of samples
        f.write(struct.pack("I", num_samples))

        # Sample values
        for value in signal.signal:
            f.write(struct.pack("d", value))


def signal_from_file(filename: str):
    """Load a signal and its metadata from a binary file."""

    with open(filename, "rb") as f:
        # Metadata
        start_time = struct.unpack("d", f.read(8))[0]
        sample_rate = struct.unpack("d", f.read(8))[0]
        duration = struct.unpack("d", f.read(8))[0]
        amplitude = struct.unpack("d", f.read(8))[0]
        period = struct.unpack("d", f.read(8))[0]

        # Signal type (UTF-8 string)
        name_len = struct.unpack("I", f.read(4))[0]
        type_name = f.read(name_len).decode("utf-8")

        # Number of samples
        num_samples = struct.unpack("I", f.read(4))[0]

        # Sample values
        values = [struct.unpack("d", f.read(8))[0] for _ in range(num_samples)]

    signal = Signal(
        signal=values,
        amplitude=amplitude,
        duration=duration,
        start_time=start_time,
        period=period,
        sample_rate=int(sample_rate)
    )

    return signal, type_name
