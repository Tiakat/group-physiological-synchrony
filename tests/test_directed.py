"""Tests for src/directed.py — lagged coupling and Granger causality."""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from directed import (downsample_1hz, lagged_xcorr, coupling_stats,
                      shifted_null_peak, granger_f)
from surrogates import empirical_p


def _driven_pair(seed=0, delay_s=10, hz=25.0, minutes=20):
    rng = np.random.default_rng(seed)
    n = int(minutes * 60 * hz)
    x = np.cumsum(rng.standard_normal(n)) * 0.01
    d = int(delay_s * hz)
    y = np.concatenate([np.zeros(d), x[:-d]]) + 0.005 * rng.standard_normal(n)
    return downsample_1hz(x, hz), downsample_1hz(y, hz)


def test_downsample_1hz():
    x = np.arange(250, dtype=float)
    assert downsample_1hz(x).shape == (10,)


def test_lagged_xcorr_recovers_delay_on_differenced():
    xd, yd = _driven_pair()
    s = coupling_stats(np.diff(xd), np.diff(yd), 60)
    assert abs(s["lead_x_s"] - 10) < 3, s  # x leads by ~10 s


def test_shifted_null_flags_real_coupling():
    xd, yd = _driven_pair()
    s = coupling_stats(np.diff(xd), np.diff(yd), 60)
    null = shifted_null_peak(np.diff(xd), np.diff(yd), 60, n=50, seed=3)
    assert empirical_p(s["peak_abs_r"], null) < 0.05


def test_shifted_null_clears_independent_pair():
    rng = np.random.default_rng(9)
    xd, _ = _driven_pair(seed=1)
    zd = downsample_1hz(np.cumsum(rng.standard_normal(30000)) * 0.01)
    s = coupling_stats(np.diff(xd), np.diff(zd), 60)
    null = shifted_null_peak(np.diff(xd), np.diff(zd), 60, n=50, seed=4)
    assert empirical_p(s["peak_abs_r"], null) > 0.05


def test_granger_finds_direction():
    xd, yd = _driven_pair()
    _, p_xy = granger_f(yd, xd)   # does x drive y?
    _, p_yx = granger_f(xd, yd)
    assert p_xy < 0.01 and p_yx > 0.05, (p_xy, p_yx)
