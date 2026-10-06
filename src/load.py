"""Load GroupAffect-4 EmotiBit physiology (LSL long format) into tidy series.

Confirmed channel map (authors' pipeline defaults + empirical verification):
  value_0  PPG (raw counts)      value_3  EDA (uS)
  value_6  device HR (bpm)       value_10 skin temperature (degC)
Sampling ~25 Hz. Files: task-{T0..T4}_run-01_acq-P{1..4}_emotibit.tsv.gz
"""
import gzip
import re
from pathlib import Path
import numpy as np
import pandas as pd

CHANNELS = {"ppg": "value_0", "eda": "value_3", "hr": "value_6", "temp": "value_10"}
TASKS = ["T0", "T1", "T2", "T3", "T4"]
FNAME_RE = re.compile(r"ses-\d+_(grp-\d+)_run\d+/physio/.*_task-(T\d)_run-\d+_acq-(P\d)_emotibit")


def iter_physio_files(raw_dir: str | Path):
    """Yield (group, task, participant, path) for T0..T4 participant files."""
    raw = Path(raw_dir)
    for p in sorted(raw.rglob("*_acq-P?_emotibit.tsv.gz")):
        m = FNAME_RE.search(p.as_posix())
        if not m:
            continue
        group, task, part = m.group(1), m.group(2), m.group(3)
        if task in TASKS:
            yield group, task, part, p


def load_participant(path: str | Path) -> pd.DataFrame:
    """Load one file -> DataFrame[time_s, eda, ppg, hr, temp], raw LSL clock."""
    with gzip.open(path, "rt") as f:
        df = pd.read_csv(f, sep="\t", usecols=["lsl_time"] + list(CHANNELS.values()))
    out = pd.DataFrame({"time_s": pd.to_numeric(df["lsl_time"], errors="coerce")})
    for name, col in CHANNELS.items():
        out[name] = pd.to_numeric(df[col], errors="coerce")
    return out.dropna(subset=["time_s"]).sort_values("time_s").reset_index(drop=True)


def load_group_task(raw_dir: str | Path, group: str, task: str,
                    target_hz: float = 25.0) -> dict[str, pd.DataFrame]:
    """Load all participants of one (group, task) onto a COMMON uniform grid.

    Returns {participant: DataFrame[time_s, eda, ppg, hr, temp]}. Grid spans the
    time overlap of all participants (LSL clock is shared within a session).
    """
    parts = {}
    for g, t, p, path in iter_physio_files(raw_dir):
        if g == group and t == task:
            parts[p] = load_participant(path)
    if len(parts) < 2:
        raise ValueError(f"fewer than 2 participants for {group}/{task}")
    t0 = max(d["time_s"].iloc[0] for d in parts.values())
    t1 = min(d["time_s"].iloc[-1] for d in parts.values())
    grid = np.arange(t0, t1, 1.0 / target_hz)
    aligned = {}
    for p, d in parts.items():
        t = d["time_s"].to_numpy()
        row = {"time_s": grid}
        for c in ("eda", "ppg", "hr", "temp"):
            row[c] = np.interp(grid, t, d[c].to_numpy())
        aligned[p] = pd.DataFrame(row)
    return aligned


def list_groups_tasks(raw_dir: str | Path) -> list[tuple[str, str]]:
    return sorted({(g, t) for g, t, _, _ in iter_physio_files(raw_dir)})
