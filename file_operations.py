import struct
from dsp_signal import Signal

def signal_to_file(signal: Signal, filename: str):
    N = len(signal.signal)

    with open(filename, "wb") as f:
        # Metadane
        f.write(struct.pack("d", signal.start_time))
        f.write(struct.pack("d", signal.sample_rate))
        f.write(struct.pack("d", signal.duration))
        f.write(struct.pack("d", signal.amplitude))
        f.write(struct.pack("d", signal.period))

        # Liczba próbek
        f.write(struct.pack("I", N))

        # Wartości sygnału
        for v in signal.signal:
            f.write(struct.pack("d", v))


def signal_from_file(filename: str) -> Signal:
    with open(filename, "rb") as f:
        # Metadane
        start_time = struct.unpack("d", f.read(8))[0]
        sample_rate = struct.unpack("d", f.read(8))[0]
        duration = struct.unpack("d", f.read(8))[0]
        amplitude = struct.unpack("d", f.read(8))[0]
        period = struct.unpack("d", f.read(8))[0]

        # Liczba próbek
        N = struct.unpack("I", f.read(4))[0]

        # Wartości sygnału
        values = []
        for _ in range(N):
            values.append(struct.unpack("d", f.read(8))[0])

    # Tworzymy sygnał poprawnie:
    return Signal(
        signal=values,
        amplitude=amplitude,
        duration=duration,
        start_time=start_time,
        period=period,
        sample_rate=int(sample_rate)
    )
