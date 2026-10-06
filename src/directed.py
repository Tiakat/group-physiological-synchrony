"""Directed (leader-follower) coupling between participants' tonic EDA.

Two convergent methods, both tested against time-shifted nulls
(circular shifts destroy coupling while preserving spectra, drift,
autocorrelation -- the right null for *directionality*):

1. lagged_xcorr: full cross-correlation function over +/- max_lag_s.
   Peak lag sign = who leads. Peak |r| vs shifted null = is it coupled?
2. granger_f: VAR-based Granger causality (OLS, F-test) in both directions.

Tonic signals are downsampled to 1 Hz first (nothing of interest above
~0.05 Hz; keeps lags interpretable in seconds and VAR estimation stable).
"""
import numpy as np
from scipy import signal as spsig
from scipy import stats


def downsample_1hz(x: np.ndarray, hz: float = 25.0) -> np.ndarray:
    x = np.asarray(x, float)
    n = int(len(x) // hz)
    return x[: n * int(hz)].reshape(n, int(hz)).mean(axis=1)


def lagged_xcorr(x: np.ndarray, y: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Cross-correlation of x,y at lags -max_lag..+max_lag (samples).

    Convention: returned `lead_x` = seconds by which x LEADS y
    (positive => x's fluctuations precede y's)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)
    n = len(x)
    corr = spsig.correlate(x, y, mode="full") / n
    lags = spsig.correlation_lags(n, n, mode="full")
    m = np.abs(lags) <= max_lag
    return lags[m], corr[m]


def coupling_stats(x: np.ndarray, y: np.ndarray, max_lag_s: int = 120) -> dict:
    """Peak coupling. `lead_x_s`: seconds by which x leads y (+ => x first).
    NOTE: on drift-dominated (tonic) signals, estimate on differenced data."""
    lags, corr = lagged_xcorr(x, y, max_lag_s)
    i = int(np.argmax(np.abs(corr)))
    return {"lead_x_s": float(-lags[i]), "peak_r": float(corr[i]),
            "peak_abs_r": float(abs(corr[i]))}


def shifted_null_peak(x: np.ndarray, y: np.ndarray, max_lag_s: int,
                      n: int = 200, min_shift_s: int = 180,
                      seed: int = 0) -> np.ndarray:
    """Null distribution of peak |r| under circular time shifts of y."""
    rng = np.random.default_rng(seed)
    min_shift_s = min(min_shift_s, len(y) // 3)
    nulls = np.empty(n)
    for i in range(n):
        shift = rng.integers(min_shift_s, len(y) - min_shift_s)
        ys = np.roll(y, shift)
        nulls[i] = coupling_stats(x, ys, max_lag_s)["peak_abs_r"]
    return nulls


def _var_design(x: np.ndarray, y: np.ndarray, order: int):
    n = len(x)
    Y = x[order:]
    cols = [np.ones(n - order)]
    for k in range(1, order + 1):
        cols.append(x[order - k: n - k])
    Xr = np.column_stack(cols)
    cols_u = cols + [y[order - k: n - k] for k in range(1, order + 1)]
    Xu = np.column_stack(cols_u)
    return Y, Xr, Xu


def granger_f(x: np.ndarray, y: np.ndarray, order: int = 5) -> tuple[float, float]:
    """Does y Granger-cause x? Returns (F, p). OLS VAR, F-test."""
    Y, Xr, Xu = _var_design(x, y, order)
    br, *_ = np.linalg.lstsq(Xr, Y, rcond=None)
    bu, *_ = np.linalg.lstsq(Xu, Y, rcond=None)
    rss_r = float(((Y - Xr @ br) ** 2).sum())
    rss_u = float(((Y - Xu @ bu) ** 2).sum())
    df1, df2 = order, len(Y) - Xu.shape[1]
    F = ((rss_r - rss_u) / df1) / (rss_u / df2) if rss_u > 0 else 0.0
    p = float(stats.f.sf(F, df1, df2))
    return F, p
