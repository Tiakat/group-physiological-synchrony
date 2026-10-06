"""Surrogate null distributions for synchrony estimates.

Two nulls, answering two different skepticisms:
1. phase_randomized: same power spectrum & autocorrelation, no true coupling.
   Kills the "it's just autocorrelation" objection.
2. shuffled_pairs: dyads formed across different groups/tasks (pseudo-dyads).
   Kills the "it's just shared task structure" objection — the Dumas &
   Fairhurst (2021) / Burgess (2013) spurious-coupling critique.

Report: observed synchrony vs null distribution -> z-score / empirical p.
"""
import numpy as np


def phase_randomized(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """IAAFT-lite: randomize Fourier phases, keep amplitudes."""
    x = np.asarray(x, float)
    X = np.fft.rfft(x)
    phases = rng.uniform(0, 2 * np.pi, size=X.shape)
    # keep DC and Nyquist phases fixed for real-valued output
    phases[0] = np.angle(X[0])
    if len(x) % 2 == 0:
        phases[-1] = np.angle(X[-1])
    return np.fft.irfft(np.abs(X) * np.exp(1j * phases), n=len(x))


def surrogate_distribution(x: np.ndarray, y: np.ndarray, stat_fn, n: int = 1000,
                           seed: int = 0) -> np.ndarray:
    """Null distribution of stat_fn under phase randomization of y."""
    rng = np.random.default_rng(seed)
    nulls = np.empty(n)
    for i in range(n):
        nulls[i] = stat_fn(x, phase_randomized(y, rng))
    return nulls


def empirical_p(observed: float, nulls: np.ndarray) -> float:
    """One-sided empirical p: P(null >= observed). NaN in -> NaN out."""
    if not np.isfinite(observed):
        return float("nan")
    nulls = nulls[np.isfinite(nulls)]
    if len(nulls) == 0:
        return float("nan")
    return (1 + np.sum(nulls >= observed)) / (1 + len(nulls))


def shuffled_dyads(signals_by_group: dict[str, dict[str, np.ndarray]],
                   rng: np.random.Generator) -> dict[str, dict[str, np.ndarray]]:
    """Build pseudo-groups by shuffling participants across real groups."""
    people = [(g, p, s) for g, d in signals_by_group.items() for p, s in d.items()]
    rng.shuffle(people)
    groups = sorted(signals_by_group)
    out, i = {}, 0
    for g in groups:
        k = len(signals_by_group[g])
        out[g] = {f"pseudo_{j}": people[i + j][2] for j in range(k)}
        i += k
    return out
