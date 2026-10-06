"""Tests for src/synchrony.py and src/surrogates.py."""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from synchrony import windowed_pearson, dyad_synchrony, group_synchrony
from surrogates import phase_randomized, empirical_p


def test_windowed_pearson_finds_coupling():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(5000)
    y = 0.8 * x + 0.2 * rng.standard_normal(5000)
    r = windowed_pearson(x, y, win=500, step=100)
    assert np.nanmean(r) > 0.7


def test_windowed_pearson_null_for_independent():
    rng = np.random.default_rng(2)
    r = windowed_pearson(rng.standard_normal(5000), rng.standard_normal(5000),
                         win=500, step=100)
    assert abs(np.nanmean(r)) < 0.1


def test_group_synchrony_averages_dyads():
    curves = {("a", "b"): np.array([0.5, 0.5]), ("a", "c"): np.array([0.1, 0.3])}
    g = group_synchrony(curves)
    assert np.allclose(g, [0.3, 0.4])


def test_dyad_synchrony_keys():
    rng = np.random.default_rng(3)
    sigs = {p: rng.standard_normal(2000) for p in ("P1", "P2", "P3")}
    d = dyad_synchrony(sigs, win=500, step=250)
    assert set(d) == {("P1", "P2"), ("P1", "P3"), ("P2", "P3")}


def test_phase_randomized_preserves_spectrum():
    rng = np.random.default_rng(4)
    # AR(1) noise: autocorrelated, no dominant sinusoid (phase shift of a
    # pure sine can stay correlated, so a sine is a bad test signal here)
    x = np.zeros(4000)
    for i in range(1, 4000):
        x[i] = 0.9 * x[i - 1] + rng.standard_normal()
    s = phase_randomized(x, rng)
    assert s.shape == x.shape
    amp_x = np.abs(np.fft.rfft(x))
    amp_s = np.abs(np.fft.rfft(s))
    assert np.corrcoef(amp_x, amp_s)[0, 1] > 0.99   # same spectrum
    assert abs(np.corrcoef(x, s)[0, 1]) < 0.2       # coupling destroyed


def test_empirical_p():
    rng = np.random.default_rng(5)
    nulls = rng.standard_normal(200)
    assert empirical_p(3.0, nulls) < 0.01
    assert empirical_p(-3.0, nulls) > 0.99
    assert np.isnan(empirical_p(np.nan, nulls))
