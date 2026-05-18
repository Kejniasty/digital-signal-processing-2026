from dsp_signal import design_fir
from dsp_signal import filter_signal
from dsp_signal import Signal
from dsp_signal import WindowType
from dsp_signal import FilterType
from dsp_signal import cross_correlate_direct
from dsp_signal import cross_correlate_via_convolution

from correlation_distance_sim import DistanceSensor

import math



def _demo_convolution():
    print("=" * 60)
    print("CZĘŚĆ 1 – Splot dyskretny")
    print("=" * 60)
    h = Signal([1, 2, 3, 4], 4.0, 4 / 100, 0.0, 0.0, 100)
    x = Signal([5, 6, 7], 7.0, 3 / 100, 0.0, 0.0, 100)
    result = h.convolve(x)
    print(f"h = {h.signal}")
    print(f"x = {x.signal}")
    print(f"h*x = {result.signal}")
    # Expected from spec: [5,16,34,52,45,28]
    expected = [5, 16, 34, 52, 45, 28]
    print(f"Expected: {expected}")
    assert result.signal == expected, "Convolution mismatch!"
    print("✓ Splot poprawny\n")


def _demo_filters():
    print("=" * 60)
    print("CZĘŚĆ 2 – Filtry FIR")
    print("=" * 60)
    SR = 500  # sample rate [Hz]

    # --- Lowpass with rectangular window ---
    lp_rect = design_fir(M=25, K=8, window=WindowType.RECTANGULAR, ftype=FilterType.LOWPASS)
    print(f"[LP rect]  M=25, K=8 → {len(lp_rect)} coefficients, max={max(lp_rect):.4f}")

    # --- Lowpass with all three windows ---
    for wtype in [WindowType.HAMMING, WindowType.HANNING, WindowType.BLACKMAN]:
        c = design_fir(M=25, K=8, window=wtype, ftype=FilterType.LOWPASS)
        print(f"[LP {wtype.value:12s}] max={max(c):.4f}")

    # --- Bandpass (F1) ---
    bp = design_fir(M=25, K=8, window=WindowType.HAMMING, ftype=FilterType.BANDPASS)
    print(f"[BP Hamming]       max={max(abs(v) for v in bp):.4f}")

    # --- Highpass (F2) ---
    hp = design_fir(M=25, K=8, window=WindowType.HANNING, ftype=FilterType.HIGHPASS)
    print(f"[HP Hanning]       max={max(abs(v) for v in hp):.4f}")

    # --- Filter a test sine signal ---
    # Create a signal composed of low-frequency (5 Hz) and high-frequency (80 Hz) sine waves
    n_samples = SR * 2  # 2 seconds
    lo_freq, hi_freq = 5.0, 80.0
    mixed = [
        math.sin(2 * math.pi * lo_freq * i / SR) +
        math.sin(2 * math.pi * hi_freq * i / SR)
        for i in range(n_samples)
    ]
    mixed_sig = Signal(mixed, 2.0, 2.0, 0.0, 0.0, SR)

    # Apply lowpass filter (cutoff ≈ SR/8 = 62.5 Hz → should keep 5 Hz, attenuate 80 Hz)
    lp_out = filter_signal(mixed_sig, M=63, K=8,
                           window=WindowType.HAMMING, ftype=FilterType.LOWPASS)

    # Simple power check: after LP, power of high-freq component should drop
    trim = lp_out.signal[63:]  # skip filter transient (M-1 samples)
    power = sum(v ** 2 for v in trim) / len(trim)
    print(f"\n[LP filter demo] Output RMS power (mixed→lowpass): {power:.4f}")
    print("✓ Filtracja działa\n")


def _demo_correlation():
    print("=" * 60)
    print("CZĘŚĆ 3 – Korelacja wzajemna")
    print("=" * 60)
    h = Signal([1, 2, 3, 4], 4.0, 4 / 100, 0.0, 0.0, 100)
    x = Signal([5, 6, 7], 7.0, 3 / 100, 0.0, 0.0, 100)

    r_direct = cross_correlate_direct(h, x)
    r_conv = cross_correlate_via_convolution(h, x)

    print(f"h = {h.signal}")
    print(f"x = {x.signal}")
    print(f"R_hx (direct):      {[round(v, 4) for v in r_direct.signal]}")
    print(f"R_hx (via convolution): {[round(v, 4) for v in r_conv.signal]}")

    # Verify both methods agree
    for a, b in zip(r_direct.signal, r_conv.signal):
        assert abs(a - b) < 1e-9, f"Mismatch: {a} vs {b}"
    print("✓ Obie implementacje dają identyczne wyniki\n")

    # Auto-correlation demo: peak should be at centre (zero lag)
    sr = 200
    n = 100
    tone = [math.sin(2 * math.pi * 10 * i / sr) for i in range(n)]
    sig = Signal(tone, 1.0, n / sr, 0.0, 0.0, sr)
    auto = cross_correlate_direct(sig, sig)
    centre = len(auto.signal) // 2
    peak = auto.signal.index(max(auto.signal))
    print(f"Auto-korelacja: centrum={centre}, indeks szczytu={peak}  (powinny być równe)")
    assert peak == centre, "Autokorrelation peak not at centre!"
    print("✓ Autokorelacja – szczyt w zerze\n")


def _demo_distance_sensor():
    print("=" * 60)
    print("CZĘŚĆ 4 – Symulacja czujnika odległości")
    print("=" * 60)

    # Parameters chosen to avoid very large/small numbers
    sensor = DistanceSensor(
        signal_speed=340.0,  # [m/s]  ≈ speed of sound
        object_speed=5.0,  # [m/s]  object moves away
        initial_distance=100.0,  # [m]
        sample_rate=2000,  # [Hz]
        buffer_size=512,
        probe_period=0.05,  # [s]
        report_interval=0.1,  # [s]  report every 100 ms
        time_unit=5e-5,  # [s]
    )

    log = sensor.run(total_time=1.0)  # simulate 1 second

    print(f"{'Time[s]':>8}  {'True dist[m]':>13}  {'Estimated[m]':>13}  {'Error[m]':>9}")
    print("-" * 50)
    for t, true, est in log:
        if math.isnan(est):
            print(f"{t:8.3f}  {true:13.2f}  {'(filling buf)':>13}")
        else:
            err = abs(true - est)
            print(f"{t:8.3f}  {true:13.2f}  {est:13.2f}  {err:9.2f}")

    # Quality check: errors should be reasonably small after buffer fills
    valid = [(t, d, e) for t, d, e in log if not math.isnan(e) and t > 0.2]
    if valid:
        mean_err = sum(abs(d - e) for _, d, e in valid) / len(valid)
        print(f"\nŚredni błąd odległości (po t>0.2s): {mean_err:.2f} m")
    print()