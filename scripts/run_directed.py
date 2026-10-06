#!/usr/bin/env python3
"""Directed coupling: who drives whom? (tonic EDA, Granger + lagged xcorr)

For every dyad in every session:
  - Granger causality both directions, each tested against 200 time-shifted
    nulls (empirical p, no parametric assumptions on autocorrelated data).
  - Lead/lag timescale from cross-correlation of *differenced* tonic
    (differencing is required: raw drift's xcorr peaks at lag 0 regardless).
Then: per-participant driver scores, and the social question -- does the
most dominant participant drive the physiology? (dominance_p1..p4 ratings).
"""
import glob
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from load import load_group_task, list_groups_tasks, TASKS
from eda import qc_ok, decompose_eda
from directed import downsample_1hz, coupling_stats, granger_f
from surrogates import empirical_p

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
FIG = ROOT / "figures"
RES = ROOT / "results"
N_NULL = 200
GRANGER_ORDER = 5  # at 1 Hz => 5 s of history


def granger_null(drv: np.ndarray, tgt: np.ndarray, order: int,
                 n: int, seed: int) -> np.ndarray:
    """Null distribution of the Granger F under circular shifts of driver."""
    rng = np.random.default_rng(seed)
    ms = min(180, len(drv) // 3)
    nulls = np.empty(n)
    for i in range(n):
        shift = rng.integers(ms, len(drv) - ms)
        nulls[i] = granger_f(tgt, np.roll(drv, shift), order)[0]
    return nulls


def dominance_table() -> pd.DataFrame:
    """Mean dominance rating per (group, task, participant)."""
    dfs = [pd.read_csv(f, sep="\t") for f in
           sorted(glob.glob(str(RAW / "beh/bids_release_no_video/*/ses-*_grp-*/beh/*stimuli_answers.tsv")))]
    df = pd.concat(dfs, ignore_index=True)
    df["grp"] = df["source_file"].str.extract(r"(grp-\d+)")
    dom = df[df["item_key"].str.match(r"dominance_p\d", na=False)].copy()
    # item_keys use lowercase p ("dominance_p1"); physio uses uppercase ("P1")
    dom["participant"] = "P" + dom["item_key"].str.extract(r"p(\d)", expand=False)
    dom["val"] = pd.to_numeric(dom["item_value"], errors="coerce")
    return dom.groupby(["grp", "task", "participant"])["val"].mean().reset_index()


def main():
    t00 = time.time()
    pairs = list_groups_tasks(RAW)
    rows, example = [], {}
    for gi, (group, task) in enumerate(pairs):
        try:
            parts = load_group_task(RAW, group, task)
        except ValueError:
            continue
        good = {p: d for p, d in parts.items() if qc_ok(d["eda"].to_numpy(), 25.0)}
        if len(good) < 2:
            print(f"skip {group} {task}"); continue
        tonic = {}
        for p, d in good.items():
            t = decompose_eda(d["eda"].to_numpy(), 25.0)["eda_tonic"].to_numpy()
            tonic[p] = downsample_1hz(t)
        ps = sorted(tonic)
        for ia in range(len(ps)):
            for ib in range(ia + 1, len(ps)):
                a, b = ps[ia], ps[ib]
                xa, xb = tonic[a], tonic[b]
                Fab = granger_f(xb, xa, GRANGER_ORDER)[0]
                Fba = granger_f(xa, xb, GRANGER_ORDER)[0]
                seed = hash((group, task, a, b)) % 2**32
                p_ab = empirical_p(Fab, granger_null(xa, xb, GRANGER_ORDER, N_NULL, seed))
                p_ba = empirical_p(Fba, granger_null(xb, xa, GRANGER_ORDER, N_NULL, seed + 1))
                dx, dy = np.diff(xa), np.diff(xb)
                lead = coupling_stats(dx, dy, 60)
                rows.append(dict(group=group, task=task, a=a, b=b,
                                 F_a_to_b=Fab, p_a_to_b=p_ab,
                                 F_b_to_a=Fba, p_b_to_a=p_ba,
                                 lead_x_s=lead["lead_x_s"], peak_r=lead["peak_r"]))
                if group == "grp-07" and task == "T1" and a == "P2" and b == "P4":
                    example = dict(Fab=Fab, p_ab=p_ab, Fba=Fba, p_ba=p_ba,
                                   null_ab=granger_null(xa, xb, GRANGER_ORDER, N_NULL, seed),
                                   dx=dx, dy=dy)
        print(f"[{gi+1}/{len(pairs)}] {group} {task}: {len(ps)} participants done")
    res = pd.DataFrame(rows)
    res.to_csv(RES / "directed_coupling.csv", index=False)
    print(f"saved {len(res)} dyads ({time.time()-t00:.0f}s)")
    summarize(res, example)


def summarize(res: pd.DataFrame, ex: dict):
    # directed dyad: one direction p<.05, the other not
    res = res.copy()
    res["a_drives"] = (res["p_a_to_b"] < 0.05) & ~(res["p_b_to_a"] < 0.05)
    res["b_drives"] = (res["p_b_to_a"] < 0.05) & ~(res["p_a_to_b"] < 0.05)
    res["directed"] = res["a_drives"] | res["b_drives"]

    # Fig 7: directed-coupling rate per task
    g = res.groupby("task").agg(rate=("directed", "mean"), n=("directed", "count")).reindex(TASKS)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(TASKS)), g["rate"], tick_label=TASKS)
    ax.set_ylabel("fraction of dyads with significant directed coupling")
    ax.set_title("Directed tonic coupling by task (Granger, shift-null p<.05)")
    for i, (r_, n_) in enumerate(zip(g["rate"], g["n"])):
        ax.text(i, r_ + 0.01, f"n={n_}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig7_directed_rate.png", dpi=150); plt.close(fig)
    print(g.round(3).to_string())

    # driver scores + dominance link
    dom = dominance_table()
    links = []
    for (group, task), d in res.groupby(["group", "task"]):
        score = {}
        for _, r_ in d.iterrows():
            score[r_["a"]] = score.get(r_["a"], 0) + int(r_["a_drives"])
            score[r_["b"]] = score.get(r_["b"], 0) + int(r_["b_drives"])
        dd = dom[(dom["grp"] == group) & (dom["task"] == task)]
        if len(dd) == 0 or len(score) < 2:
            continue
        dmap = dict(zip(dd["participant"], dd["val"]))
        common = [p for p in score if p in dmap]
        if len(common) < 2:
            continue
        top_dom = max(common, key=lambda p: dmap[p])
        top_drv = max(common, key=lambda p: score[p])
        links.append({"group": group, "task": task,
                      "n": len(common),
                      "top_dominant": top_dom, "top_driver": top_drv,
                      "match": int(top_dom == top_drv),
                      "spearman": sps.spearmanr([dmap[p] for p in common],
                                                [score[p] for p in common])[0]})
    links = pd.DataFrame(links)
    if len(links) == 0:
        print("\nno group-task with both dominance ratings and driver scores; skipping dominance link")
    else:
        links.to_csv(RES / "dominance_driver.csv", index=False)
        m = links["match"].sum(); n = len(links)
        # binomial: P(top dominant == top driver) vs 1/4 chance
        pval = sps.binomtest(m, n, 0.25, alternative="greater").pvalue
        print(f"\ndominant==driver: {m}/{n} (chance 25%), binomial p={pval:.3f}")
        print(f"median dominance-driver spearman: {links['spearman'].median():.3f}")

        # Fig 8: dominance vs driver score
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.scatter(links["spearman"], links["match"], alpha=0.6)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("dominance-driver rank correlation (per group-task)")
        ax.set_ylabel("top dominant == top driver")
        ax.set_yticks([0, 1]); ax.set_yticklabels(["no", "yes"])
        ax.set_title("Does the most dominant member drive physiology?")
        fig.tight_layout(); fig.savefig(FIG / "fig8_dominance_driver.png", dpi=150); plt.close(fig)

    # Fig 6: example dyad
    if ex:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].hist(ex["null_ab"], bins=30, alpha=0.7, label="shift null (P2->P4)")
        ax[0].axvline(ex["Fab"], color="red", lw=2, label=f'observed F={ex["Fab"]:.1f}, p={ex["p_ab"]:.3f}')
        ax[0].set_xlabel("Granger F"); ax[0].set_ylabel("count"); ax[0].legend()
        ax[0].set_title("Is the directed coupling real? (grp-07 T1, P2->P4)")
        lags = np.arange(-60, 61)
        from directed import lagged_xcorr
        _, cc = lagged_xcorr(ex["dx"], ex["dy"], 60)
        ax[1].plot(-lags, cc, lw=1.2)
        ax[1].axvline(0, color="k", lw=0.8); ax[1].set_xlabel("lead of P2 over P4 (s)")
        ax[1].set_ylabel("xcorr (differenced tonic)")
        ax[1].set_title("Lead/lag timescale")
        fig.tight_layout(); fig.savefig(FIG / "fig6_directed_example.png", dpi=150); plt.close(fig)
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
