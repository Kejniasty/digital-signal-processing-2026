from dsp_signal import Signal, dit_fft, dwt_one_level, idwt_one_level

# fft demo
sig = Signal(
    [1, 2, 3, 4, 5, 6, 7, 8],
    amplitude=8,
    duration=1.0,
    sample_rate=8
)

spectrum = dit_fft(sig)

print("Magnitude:")
print(spectrum.magnitude())

spectrum.plot_w2()

# dwt demo
approx, detail = dwt_one_level(sig, "db4")

print("Approximation:", approx.signal)
print("Detail:", detail.signal)

# reconstruction demo

approx, detail = dwt_one_level(sig, "db4")

reconstructed = idwt_one_level(
    approx,
    detail,
    "db4"
)

print("Original:     ", sig.signal)
print("Reconstructed:", reconstructed.signal)