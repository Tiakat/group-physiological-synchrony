#!/usr/bin/env python3
"""Full analysis: tonic/phasic EDA synchrony in interacting groups (GroupAffect-4).

Runs end to end: load -> QC -> decompose -> synchrony -> surrogates ->
task contrasts -> prediction -> figures + results CSV.

Usage:  python scripts/run_analysis.py [--quick]
  --quick  only 2 groups (smoke test)
"""
import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal as spsig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from load import load_group_task, list_groups_tasks, TASKS
from eda import qc_ok, decompose_eda
from synchrony import windowed_pearson, dyad_synchrony, group_synchrony
from surrogates import phase_randomized, empirical_p

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
FIG = ROOT / "figures"
RES = ROOT / "results"
WIN_S, STEP_S, HZ = 60, 10, 25.0
N_NULL = 200


def main(quick: bool = False):
    t00 = time.time()
    FIG.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
    pairs = list_groups_tasks(RAW)
    if quick:
        pairs = pairs[:4]
    print(f"{len(pairs)} group-task sessions")

    rows = []
    example = {}
    for gi, (group, task) in enumerate(pairs):
        try:
            parts = load_group_task(RAW, group, task)
        except ValueError as e:
            print("skip", group, task, e); continue
        good = {p: d for p, d in parts.items() if qc_ok(d["eda"].to_numpy())}
        if len(good) < 2:
            print("skip", group, task, "QC"); continue
        comp = {}
        for p, d in good.items():
            eda = d["eda"].to_numpy()
            try:
                comp[p] = decompose_eda(eda, HZ)
            except Exception as e:
                print("decompose failed", group, task, p, e)
        if len(comp) < 2:
            print("skip", group, task, "decompose"); continue
        win, step = int(WIN_S * HZ), int(STEP_S * HZ)
        tonic = {p: c["eda_tonic"].to_numpy() for p, c in comp.items()}
        phasic = {p: c["eda_phasic"].to_numpy() for p, c in comp.items()}
        # control: linearly detrended tonic (is resting synchrony just co-drift?)
        tonic_dt = {p: spsig.detrend(t) for p, t in tonic.items()}
        tc = dyad_synchrony(tonic, win, step)
        pc = dyad_synchrony(phasic, win, step)
        dc = dyad_synchrony(tonic_dt, win, step)
        g_tonic, g_phasic = group_synchrony(tc), group_synchrony(pc)
        g_tonic_dt = group_synchrony(dc)
        m_tonic = float(np.nanmean(g_tonic)); m_phasic = float(np.nanmean(g_phasic))
        m_tonic_dt = float(np.nanmean(g_tonic_dt))

        # surrogate nulls on the group-mean synchrony (phase-randomized)
        rng = np.random.default_rng(hash((group, task)) % 2**32)
        null_t, null_p, null_d = np.empty(N_NULL), np.empty(N_NULL), np.empty(N_NULL)
        keys = sorted(tonic)
        for i in range(N_NULL):
            surr = {k: phase_randomized(tonic[k], rng) for k in keys}
            null_t[i] = np.nanmean(group_synchrony(dyad_synchrony(surr, win, step)))
            surrp = {k: phase_randomized(phasic[k], rng) for k in keys}
            null_p[i] = np.nanmean(group_synchrony(dyad_synchrony(surrp, win, step)))
            surrd = {k: phase_randomized(tonic_dt[k], rng) for k in keys}
            null_d[i] = np.nanmean(group_synchrony(dyad_synchrony(surrd, win, step)))
        p_tonic = empirical_p(m_tonic, null_t)
        p_phasic = empirical_p(m_phasic, null_p)
        p_tonic_dt = empirical_p(m_tonic_dt, null_d)
        rows.append(dict(group=group, task=task, n_participants=len(comp),
                         mean_tonic_sync=m_tonic, mean_phasic_sync=m_phasic,
                         mean_tonic_detrended_sync=m_tonic_dt,
                         p_tonic=p_tonic, p_phasic=p_phasic,
                         p_tonic_detrended=p_tonic_dt))
        if group == "grp-07" and task == "T1":
            example = dict(eda=good["P1"]["eda"].to_numpy()[: int(300 * HZ)],
                           tonic=tonic["P1"][: int(300 * HZ)],
                           phasic=phasic["P1"][: int(300 * HZ)],
                           g_tonic=g_tonic, g_phasic=g_phasic,
                           null_t=null_t, m_tonic=m_tonic)
        print(f"[{gi+1}/{len(pairs)}] {group} {task}: tonic r={m_tonic:.3f} (p={p_tonic:.3f})  "
              f"phasic r={m_phasic:.3f} (p={p_phasic:.3f})  tonic_detr r={m_tonic_dt:.3f} (p={p_tonic_dt:.3f})")

    res = pd.DataFrame(rows)
    res.to_csv(RES / "synchrony_summary.csv", index=False)
    print("results ->", RES / "synchrony_summary.csv", f"({time.time()-t00:.0f}s)")
    make_figures(res, example)
    task_contrast(res)


def make_figures(res: pd.DataFrame, ex: dict):
    # Fig 1: example tonic/phasic decomposition
    if ex:
        fig, ax = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        t = np.arange(len(ex["eda"])) / HZ
        ax[0].plot(t, ex["eda"], lw=0.6); ax[0].set_ylabel("EDA (µS)"); ax[0].set_title("Example participant (grp-07, T1): raw EDA and its decomposition")
        ax[1].plot(t, ex["tonic"], lw=0.8, color="C1"); ax[1].set_ylabel("tonic (µS)")
        ax[2].plot(t, ex["phasic"], lw=0.6, color="C2"); ax[2].set_ylabel("phasic (µS)"); ax[2].set_xlabel("time (s)")
        fig.tight_layout(); fig.savefig(FIG / "fig1_example_decomposition.png", dpi=150); plt.close(fig)
        # Fig 2: group synchrony curves, tonic vs phasic
        fig, ax = plt.subplots(figsize=(10, 3.5))
        tt = np.arange(len(ex["g_tonic"])) * STEP_S / 60
        ax.plot(tt, ex["g_tonic"], label="tonic synchrony", lw=1.5)
        ax.plot(tt, ex["g_phasic"], label="phasic synchrony", lw=1.5)
        ax.set_xlabel("time (min)"); ax.set_ylabel("group synchrony (r)")
        ax.set_title("Group synchrony over time, grp-07 T1 (hidden-profile decision)")
        ax.legend(); fig.tight_layout(); fig.savefig(FIG / "fig2_synchrony_curves.png", dpi=150); plt.close(fig)
        # Fig 4: surrogate null
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(ex["null_t"], bins=30, alpha=0.7, label="phase-randomized null")
        ax.axvline(ex["m_tonic"], color="red", lw=2, label=f"observed r={ex['m_tonic']:.3f}")
        ax.set_xlabel("group tonic synchrony"); ax.set_ylabel("count")
        ax.set_title("Surrogate test: is the synchrony real?"); ax.legend()
        fig.tight_layout(); fig.savefig(FIG / "fig4_surrogate_null.png", dpi=150); plt.close(fig)


def task_contrast(res: pd.DataFrame):
    # Fig 3: mean synchrony per task, tonic vs phasic, with significance stars
    # (drop sessions with no valid synchrony, e.g. whole-group sensor dropout)
    res = res[np.isfinite(res["mean_tonic_sync"]) & np.isfinite(res["mean_phasic_sync"])]
    g = res.groupby("task").agg(mt=("mean_tonic_sync", "mean"),
                                mp=("mean_phasic_sync", "mean"),
                                md=("mean_tonic_detrended_sync", "mean"),
                                st=("mean_tonic_sync", "sem"),
                                sp=("mean_phasic_sync", "sem"),
                                sd=("mean_tonic_detrended_sync", "sem")).reindex(TASKS)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(TASKS)); w = 0.25
    st = g["st"].fillna(0).to_numpy(); sp = g["sp"].fillna(0).to_numpy(); sd = g["sd"].fillna(0).to_numpy()
    ax.bar(x - w, g["mt"], w, yerr=st, label="tonic", capsize=3)
    ax.bar(x, g["md"], w, yerr=sd, label="tonic (detrended)", capsize=3)
    ax.bar(x + w, g["mp"], w, yerr=sp, label="phasic", capsize=3)
    for i, t in enumerate(TASKS):
        sub = res[res["task"] == t]
        if len(sub) and (sub["p_tonic"] < 0.05).mean() > 0.5:
            ax.text(i - w, g["mt"].iloc[i] + st[i] + 0.005, "*", ha="center", fontsize=14)
    ax.set_xticks(x); ax.set_xticklabels(TASKS)
    ax.set_ylabel("mean group synchrony (r ± SEM)")
    ax.set_title("Tonic vs phasic synchrony by task (T0 = resting baseline)")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "fig3_task_contrast.png", dpi=150); plt.close(fig)
    print("figures ->", FIG)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
