"""EDA quality control, resampling, and tonic/phasic decomposition.

Decomposition (transparent, dependency-light):
  tonic  = 4th-order Butterworth low-pass at 0.05 Hz  (slow drift: shared context)
  phasic = raw - tonic, high-passed at 0.05 Hz         (event responses)
This mirrors the standard tonic/phasic split (cf. Benedek & Kaernbach 2010;
the authors' own pipeline uses the same conceptual split).
"""
import numpy as np
import pandas as pd
from scipy import signal as spsig


RAIL_VALUE = 0.030  # EmotiBit pins EDA here on electrode dropout


def _run_fractions(eda: np.ndarray, min_run_s: float, hz: float) -> float:
    """Fraction of samples inside constant runs lasting >= min_run_s."""
    dif = np.abs(np.diff(eda)) < 1e-12
    if not dif.any():
        return 0.0
    # lengths of True runs in dif
    edges = np.diff(np.concatenate([[False], dif, [False]]).astype(int))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    long = sum(e - s + 1 for s, e in zip(starts, ends)
               if (e - s + 1) >= min_run_s * hz)
    return long / len(eda)


def qc_eda(eda: np.ndarray, sampling_rate: float) -> dict:
    """QC flags. Notes:
    - The EmotiBit repeats EDA values across LSL rows (~60% exact-zero
      diffs at 25 Hz read rate); short repeats are NORMAL quantization, not
      dropout. Only long constant runs count as 'stuck'.
    - On electrode dropout the device rails at exactly 0.030 uS."""
    eda = np.asarray(eda, dtype=float)
    finite = np.isfinite(eda)
    missing = float(1 - finite.mean())
    ef = eda[finite]
    rail = float(np.mean(np.abs(ef - RAIL_VALUE) < 1e-9)) if len(ef) else 0.0
    stuck = _run_fractions(ef, min_run_s=10.0, hz=sampling_rate) if len(ef) else 1.0
    oor = float(np.mean((ef < 0.01) | (ef > 100.0))) if len(ef) else 1.0
    return {"missing_fraction": missing, "rail_fraction": rail,
            "stuck_fraction": stuck, "out_of_range_fraction": oor,
            "n": len(eda), "sampling_rate": sampling_rate}


def qc_ok(eda: np.ndarray, sampling_rate: float = 25.0) -> bool:
    q = qc_eda(eda, sampling_rate)
    return (q["missing_fraction"] < 0.3 and q["rail_fraction"] < 0.3
            and q["stuck_fraction"] < 0.5 and q["out_of_range_fraction"] < 0.5)


def resample_uniform(df: pd.DataFrame, target_hz: float = 25.0) -> pd.DataFrame:
    """Resample to a uniform grid (EmotiBit is ~25 Hz but can jitter)."""
    t = df["time_s"].to_numpy()
    t_uniform = np.arange(t.min(), t.max(), 1.0 / target_hz)
    out = {"time_s": t_uniform}
    for col in df.columns:
        if col == "time_s":
            continue
        out[col] = np.interp(t_uniform, t, pd.to_numeric(df[col], errors="coerce").to_numpy())
    return pd.DataFrame(out)


def decompose_eda(eda: np.ndarray, sampling_rate: float,
                  tonic_cutoff: float = 0.05) -> pd.DataFrame:
    """Return DataFrame with tonic and phasic components (uS).

    Second-order-sections filtering (numerically stable at very low cutoffs).
    """
    eda = np.asarray(eda, dtype=float)
    # guard: interpolate short NaN gaps, else filtering propagates NaN
    mask = np.isnan(eda)
    if mask.any():
        idx = np.arange(len(eda))
        eda = eda.copy()
        eda[mask] = np.interp(idx[mask], idx[~mask], eda[~mask])
    nyq = sampling_rate / 2.0
    sos_lp = spsig.butter(4, tonic_cutoff / nyq, btype="low", output="sos")
    sos_hp = spsig.butter(4, tonic_cutoff / nyq, btype="high", output="sos")
    tonic = spsig.sosfiltfilt(sos_lp, eda)
    phasic = spsig.sosfiltfilt(sos_hp, eda - tonic)
    return pd.DataFrame({"eda_tonic": tonic, "eda_phasic": phasic})


def scr_peaks(phasic: np.ndarray, sampling_rate: float,
              min_height: float = 0.01, min_dist_s: float = 1.0) -> np.ndarray:
    """Indices of phasic SCR peaks (simple amplitude threshold)."""
    peaks, _ = spsig.find_peaks(phasic, height=min_height,
                                distance=int(min_dist_s * sampling_rate))
    return peaks
