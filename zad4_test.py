# test_transforms.py

import math
import numpy as np
import pytest

from dsp_signal import (
    Signal,
    dft,
    idft,
    dit_fft,
    idit_fft,
    dwt_one_level,
    idwt_one_level, dif_fft, dct2, idct2, fct2, walsh_hadamard, fast_walsh_hadamard,
)

# auxiliary
EPS = 1e-6

def assert_lists_close(a, b, eps=EPS):
    assert len(a) == len(b)

    for x, y in zip(a, b):
        assert abs(x - y) < eps

# -----------------------
# Fourier Transform Tests
# -----------------------

def test_dft_idft_roundtrip():
    samples = [1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, -1.0]
    sig = Signal(samples, amplitude=max(map(abs, samples)),
                 duration=1.0, sample_rate=8)

    spectrum = dft(sig)
    reconstructed = idft(spectrum)

    assert np.allclose(
        reconstructed.signal,
        sig.signal,
        atol=1e-10
    )


def test_fft_matches_dft():
    samples = [1, 2, 3, 4, 5, 6, 7, 8]
    sig = Signal(samples, amplitude=8,
                 duration=1.0, sample_rate=8)

    X_dft = dft(sig)
    X_fft = dit_fft(sig)

    assert np.allclose(
        X_dft.signal,
        X_fft.signal,
        atol=1e-10
    )


def test_fft_inverse_roundtrip():
    samples = [0, 1, 0, -1, 0, 1, 0, -1]
    sig = Signal(samples, amplitude=1,
                 duration=1.0, sample_rate=8)

    spectrum = dit_fft(sig)
    reconstructed = idit_fft(spectrum)

    assert np.allclose(
        reconstructed.signal,
        sig.signal,
        atol=1e-10
    )


def test_single_tone_peak():
    N = 8
    k = 1

    samples = [
        math.sin(2 * math.pi * k * n / N)
        for n in range(N)
    ]

    sig = Signal(samples,
                 amplitude=1.0,
                 duration=1.0,
                 sample_rate=N)

    spectrum = dit_fft(sig)

    magnitudes = [abs(v) for v in spectrum.signal]

    peak_bin = np.argmax(magnitudes)

    assert peak_bin in (1, N - 1)


def test_non_power_of_two_rejected():
    samples = [1, 2, 3, 4, 5]

    sig = Signal(samples,
                 amplitude=5,
                 duration=1.0,
                 sample_rate=5)

    import pytest

    with pytest.raises(ValueError):
        dft(sig)

    with pytest.raises(ValueError):
        dit_fft(sig)


# ---------------------
# Wavelet Transform Tests
# ---------------------

@pytest.mark.parametrize("wavelet", ["db4", "db6", "db8"])
def test_dwt_idwt_roundtrip(wavelet):
    samples = [1, 2, 3, 4, 5, 6, 7, 8]
    sig = Signal(samples,
                 amplitude=8,
                 duration=1.0,
                 sample_rate=8)

    approx, detail = dwt_one_level(sig, wavelet)

    reconstructed = idwt_one_level(
        approx,
        detail,
        wavelet
    )

    assert np.allclose(
        reconstructed.signal,
        sig.signal,
        atol=1e-10
    )


@pytest.mark.parametrize("wavelet", ["db4", "db6", "db8"])
def test_dwt_halves_length(wavelet):
    samples = list(range(16))

    sig = Signal(samples,
                 amplitude=15,
                 duration=1.0,
                 sample_rate=16)

    approx, detail = dwt_one_level(sig, wavelet)

    assert len(approx.signal) == 8
    assert len(detail.signal) == 8


def test_dwt_preserves_constant_signal_in_approximation():
    samples = [5.0] * 8

    sig = Signal(samples,
                 amplitude=5,
                 duration=1.0,
                 sample_rate=8)

    approx, detail = dwt_one_level(sig, "db4")

    detail_energy = sum(x * x for x in detail.signal)

    assert detail_energy < 1e-10


def test_invalid_wavelet():
    import pytest

    sig = Signal(
        [1, 2, 3, 4, 5, 6, 7, 8],
        amplitude=8,
        duration=1.0,
        sample_rate=8
    )

    with pytest.raises(ValueError):
        dwt_one_level(sig, "unknown")

def test_dit_fft_matches_dft():

    sig = Signal(
        signal=[1, 2, 3, 4, 5, 6, 7, 8],
        sample_rate=8
    )

    X_dft = dft(sig)
    X_fft = dit_fft(sig)

    assert len(X_dft.signal) == len(X_fft.signal)

    for a, b in zip(X_dft.signal, X_fft.signal):
        assert abs(a - b) < EPS

def test_dif_fft_matches_dft():

    sig = Signal(
        signal=[1, 2, 3, 4, 5, 6, 7, 8],
        sample_rate=8
    )

    X_dft = dft(sig)
    X_dif = dif_fft(sig)

    for a, b in zip(X_dft.signal, X_dif.signal):
        assert abs(a - b) < EPS

def test_fft_inverse_recovers_signal():

    data = [1, 2, 3, 4, 5, 6, 7, 8]

    sig = Signal(
        signal=data,
        sample_rate=8
    )

    spectrum = dit_fft(sig)

    reconstructed = idit_fft(spectrum)

    assert_lists_close(
        reconstructed.signal,
        data
    )

def test_dft_inverse_recovers_signal():

    data = [0.5, 1.5, -2.0, 4.0]

    sig = Signal(
        signal=data,
        sample_rate=4
    )

    spectrum = dft(sig)

    reconstructed = idft(spectrum)

    assert_lists_close(
        reconstructed.signal,
        data
    )

def test_dct_constant_signal():

    sig = Signal(
        signal=[1, 1, 1, 1],
        sample_rate=4
    )

    X = dct2(sig)

    assert abs(X.signal[0]) > 0.1

    for v in X.signal[1:]:
        assert abs(v) < EPS

def test_dct_inverse():

    data = [1, 2, 3, 4]

    sig = Signal(
        signal=data,
        sample_rate=4
    )

    coeffs = dct2(sig)

    reconstructed = idct2(coeffs)

    assert_lists_close(
        reconstructed.signal,
        data
    )

def test_fct_matches_dct():

    sig = Signal(
        signal=[1, 2, 3, 4, 5, 6, 7, 8],
        sample_rate=8
    )

    dct = dct2(sig)
    fct = fct2(sig)

    assert_lists_close(
        dct.signal,
        fct.signal
    )

def test_fwht_matches_wht():

    sig = Signal(
        signal=[1, 2, 3, 4, 5, 6, 7, 8],
        sample_rate=8
    )

    wht = walsh_hadamard(sig)
    fwht = fast_walsh_hadamard(sig)

    assert_lists_close(
        wht.signal,
        fwht.signal
    )

def test_fwht_inverse():

    data = [1, 2, 3, 4, 5, 6, 7, 8]

    sig = Signal(
        signal=data,
        sample_rate=8
    )

    transformed = fast_walsh_hadamard(sig)

    reconstructed = fast_walsh_hadamard(transformed)

    assert_lists_close(
        reconstructed.signal,
        data
    )

@pytest.mark.parametrize(
    "n",
    [2, 4, 8, 16, 32]
)
def test_transform_preserves_length(n):

    sig = Signal(
        signal=list(range(n)),
        sample_rate=n
    )

    assert len(dif_fft(sig).signal) == n
    assert len(dit_fft(sig).signal) == n
    assert len(dct2(sig).signal) == n
    assert len(fct2(sig).signal) == n
    assert len(walsh_hadamard(sig).signal) == n
    assert len(fast_walsh_hadamard(sig).signal) == n


def test_non_power_of_two_raises():

    sig = Signal(
        signal=[1, 2, 3, 4, 5],
        sample_rate=5
    )

    with pytest.raises(ValueError):
        dit_fft(sig)

    with pytest.raises(ValueError):
        dif_fft(sig)

    with pytest.raises(ValueError):
        dct2(sig)

    with pytest.raises(ValueError):
        fct2(sig)

    with pytest.raises(ValueError):
        walsh_hadamard(sig)

    with pytest.raises(ValueError):
        fast_walsh_hadamard(sig)
