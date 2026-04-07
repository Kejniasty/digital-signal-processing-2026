from dsp_signal import Signal


def signal_to_file(signal: "Signal", filename: str):
    file = open(filename, "w")
    file.write(str(signal))
    file.close()
    return True

def signal_from_file(filename: str):
    file = open(filename, "r")
    signal = Signal()
    signal.from_string(file.read())
    file.close()
    return signal