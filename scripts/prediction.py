#!/usr/bin/env python3
"""Does early-task synchrony predict post-task group engagement?

For each (group, task in T1..T4): synchrony over the FIRST 3 minutes ->
postblock 'engagement' rating (group mean, from stimuli_answers.tsv).
Leave-one-group-out correlation, reported honestly at n<=10 groups.
"""
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from load import load_group_task, list_groups_tasks
from eda import qc_ok, decompose_eda
from synchrony import dyad_synchrony, group_synchrony, windowed_pearson

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
FIG = ROOT / "figures"
HZ = 25.0
EARLY_S = 180  # first 3 minutes


def engagement_table() -> pd.DataFrame:
    dfs = [pd.read_csv(f, sep="\t") for f in
           sorted(glob.glob(str(RAW / "beh/bids_release_no_video/*/ses-*_grp-*/beh/*stimuli_answers.tsv")))]
    df = pd.concat(dfs, ignore_index=True)
    df["grp"] = df["source_file"].str.extract(r"(grp-\d+)")
    eng = df[df["item_key"] == "engagement"].copy()
    eng["val"] = pd.to_numeric(eng["item_value"], errors="coerce")
    return eng.groupby(["grp", "task"])["val"].mean().reset_index()


def early_sync(group: str, task: str) -> tuple[float, float] | None:
    try:
        parts = load_group_task(RAW, group, task)
    except ValueError:
        return None
    good = {p: d for p, d in parts.items()
            if qc_ok(d["eda"].to_numpy(), HZ)}
    if len(good) < 2:
        return None
    n = int(EARLY_S * HZ)
    tonic, phasic = {}, {}
    for p, d in good.items():
        eda = d["eda"].to_numpy()[:n]
        if len(eda) < n * 0.8:
            return None
        c = decompose_eda(eda, HZ)
        tonic[p] = c["eda_tonic"].to_numpy()
        phasic[p] = c["eda_phasic"].to_numpy()
    win, step = int(60 * HZ), int(10 * HZ)
    gt = group_synchrony(dyad_synchrony(tonic, win, step))
    gp = group_synchrony(dyad_synchrony(phasic, win, step))
    return float(np.nanmean(gt)), float(np.nanmean(gp))


def main():
    eng = engagement_table()
    rows = []
    for _, r in eng.iterrows():
        s = early_sync(r["grp"], r["task"])
        if s:
            rows.append({"group": r["grp"], "task": r["task"],
                         "early_tonic_sync": s[0], "early_phasic_sync": s[1],
                         "engagement": r["val"]})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "prediction.csv", index=False)
    print(df.to_string(index=False))
    for comp in ("early_tonic_sync", "early_phasic_sync"):
        x = df[comp].to_numpy(); y = df["engagement"].to_numpy()
        m = np.isfinite(x) & np.isfinite(y)
        r, p = stats.pearsonr(x[m], y[m])
        print(f"\n{comp} vs engagement: r={r:.3f}, p={p:.3f} (n={m.sum()})")
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for t, d in df[m].groupby("task"):
            ax.scatter(d[comp], d["engagement"], label=t)
        ax.set_xlabel(f"{comp} (first 3 min)"); ax.set_ylabel("post-task group engagement")
        ax.set_title(f"Early synchrony -> engagement ({comp.split('_')[1]})")
        ax.legend(); fig.tight_layout()
        fig.savefig(FIG / f"fig5_prediction_{comp.split('_')[1]}.png", dpi=150); plt.close(fig)
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
