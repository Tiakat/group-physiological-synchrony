# Tonic vs phasic EDA synchrony in interacting groups

Do **tonic** and **phasic** electrodermal activity synchronize to different
social features during four-person interaction?

Built on the public [GroupAffect-4 dataset](https://zenodo.org/records/20809799)
(10 groups × 4 participants, EmotiBit EDA/HR/temperature at ~25 Hz, tasks
T0 resting → T1 hidden-profile decision → T2 negotiation → T3 idea generation
→ T4 public-goods game).

## What this repo does

1. **Loads** raw LSL long-format EmotiBit files and aligns all four
   participants of each group × task onto a common uniform time grid
   (`src/load.py`).
2. **QC + decomposition** (`src/eda.py`): rail-aware quality control (the
   EmotiBit pins at exactly 0.030 µS on electrode dropout — treated as
   missing, not physiology), then a tonic/phasic split via zero-phase
   Butterworth filtering (tonic < 0.05 Hz; phasic = remainder).
3. **Synchrony** (`src/synchrony.py`): windowed Pearson correlation
   (60 s windows, 10 s steps) per dyad, averaged to a group synchrony curve —
   computed separately for tonic and phasic signals.
4. **Surrogate testing** (`src/surrogates.py`): every group × task synchrony
   estimate is tested against 200 phase-randomized nulls (same spectrum and
   autocorrelation, coupling destroyed). A linearly-detrended tonic control
   checks whether resting synchrony is just shared drift.
5. **Task contrasts**: T0 resting baseline vs the four active tasks.
6. **Prediction**: does early-task (first 3 min) synchrony predict post-task
   group engagement ratings?
7. **Directed coupling** (`src/directed.py`, `scripts/run_directed.py`):
   Granger causality on tonic EDA in both directions per dyad, each tested
   against 200 time-shifted nulls; lead/lag timescale from cross-correlation
   of *differenced* tonic (required: raw drift's xcorr peaks at lag 0).
   Per-participant driver scores, linked to peer dominance ratings
   (`dominance_p1..p4`).

## Reproduce

```bash
pip install -r requirements.txt
bash data/download_groupaffect4.sh   # fetch public data (~190 MB zips)
python scripts/run_analysis.py       # synchrony pipeline -> results/ + figures/
python scripts/prediction.py         # early synchrony -> engagement
python scripts/run_directed.py       # directed coupling -> results/ + figures/
```

Or run the notebook top to bottom: `notebooks/01_eda_synchrony.ipynb`.

## Key findings (see notebook for figures)

- **Tonic synchrony is a resting phenomenon, not an interaction one.**
  Mean group tonic synchrony at rest (T0): r = 0.11, significant in 3/7
  groups; during the four interaction tasks it collapses to ≈ 0.
  The resting "baseline" is the condition of *maximal* shared context —
  which flips the usual hyperscanning logic (task > rest = social coupling).
- **Much of it is shared drift**: linear detrending drops T0 tonic
  synchrony from 0.11 to 0.05. Phase-randomized nulls don't control for
  co-trending, so the detrended control is the honest estimate.
- **Phasic synchrony is ≈ 0 everywhere** — fast sympathetic responses don't
  align across members at 60 s resolution. A boundary condition for "shared
  arousal" claims.
- **Early phasic synchrony predicts engagement**: first-3-minutes phasic
  synchrony vs post-task group engagement gives r = 0.61, p = .004
  (Spearman r = 0.49, p = .030; leave-one-out r ∈ [0.47, 0.66]; exploratory,
  20 group × task observations from 7 groups).
- **Directed coupling exists but is weak and context-free.** ~21% of dyads
  show significant directed tonic coupling (Granger, time-shifted nulls)
  vs ~10% expected under the null — but rates are flat across tasks
  (16–27%, rest included), with no stable leader-follower timescale
  (median driver lead 6 s, IQR spans zero).
- **Dominance doesn't predict driving.** The most dominant member is the
  top physiological driver in 4/27 group-tasks (chance 25%, binomial
  p = .93). The obvious social hypothesis fails cleanly — reported as a
  null, not buried.

## Limitations

- EmotiBit `value_*` channel indices follow the dataset authors' pipeline
  defaults, confirmed empirically (EDA-like statistics on `value_3`,
  HR-like on `value_6`, temperature-like on `value_10`); the mapping is
  documented, not manufacturer-certified.
- ~15% of group × task sessions lost to sensor dropout (whole-group or
  majority-participant rail/flat EDA), in line with the authors' reported
  ~71% EDA availability.
- Phase-randomized nulls control for autocorrelation but not for shared
  linear drift — hence the detrended control.

## Layout

```
src/            load.py, eda.py, synchrony.py, surrogates.py, directed.py
scripts/        run_analysis.py, prediction.py, run_directed.py,
                execute_notebook.py
notebooks/      01_eda_synchrony.ipynb   (executed, with outputs)
data/           download_groupaffect4.sh (raw data not committed)
results/        synchrony_summary.csv, prediction.csv,
                directed_coupling.csv, dominance_driver.csv
figures/        publication figures (PNG)
tests/          unit tests for the core functions
```
