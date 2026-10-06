"""Pairwise and group-level synchrony metrics for slow physiological signals.

Convention: every metric is computed per dyad, per window; group synchrony is
the mean over the 6 dyads of a 4-person group. Significance always comes from
src/surrogates.py, never from the raw value alone.
"""
import itertools
import numpy as np
from scipy import stats
from numpy.lib.stride_tricks import sliding_window_view


def windowed_pearson(x: np.ndarray, y: np.ndarray, win: int, step: int) -> np.ndarray:
    """Pearson r in sliding windows (vectorized). Returns array of r values."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    if n < win:
        return np.array([np.nan])
    xw = sliding_window_view(x, win)[::step]
    yw = sliding_window_view(y, win)[::step]
    xm = xw.mean(axis=1, keepdims=True)
    ym = yw.mean(axis=1, keepdims=True)
    num = ((xw - xm) * (yw - ym)).sum(axis=1)
    den = np.sqrt(((xw - xm) ** 2).sum(axis=1) * ((yw - ym) ** 2).sum(axis=1))
    out = num / den
    out[den < 1e-12] = np.nan
    return out


def dyad_synchrony(signals: dict[str, np.ndarray], win: int, step: int,
                   metric=windowed_pearson) -> dict[tuple[str, str], np.ndarray]:
    """Synchrony time-course for every dyad. signals: participant -> 1D array."""
    out = {}
    for a, b in itertools.combinations(sorted(signals), 2):
        out[(a, b)] = metric(signals[a], signals[b], win, step)
    return out


def group_synchrony(dyad_curves: dict, agg=np.nanmean) -> np.ndarray:
    """Collapse dyad curves to one group-level synchrony time-course."""
    mat = np.vstack(list(dyad_curves.values()))
    return agg(mat, axis=0)


def lagged_synchrony(x: np.ndarray, y: np.ndarray, max_lag: int, win: int) -> tuple[np.ndarray, np.ndarray]:
    """Cross-correlation over lags within one window (for the cascade question:
    which signal leads?). Returns (lags, r)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = min(len(x), len(y), win)
    x, y = x[:n] - x[:n].mean(), y[:n] - y[:n].mean()
    lags = np.arange(-max_lag, max_lag + 1)
    rs = [stats.pearsonr(x[max(0, -l): n - max(0, l)],
                         y[max(0, l): n - max(0, -l)])[0] if n - abs(l) > 10 else np.nan
          for l in lags]
    return lags, np.array(rs)
