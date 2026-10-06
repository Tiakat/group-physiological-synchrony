"""Tests for src/eda.py — QC flags and tonic/phasic decomposition."""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from eda import qc_eda, qc_ok, decompose_eda

HZ = 25.0


def test_qc_ok_on_clean_signal():
    rng = np.random.default_rng(0)
    eda = 0.15 + 0.05 * np.sin(np.linspace(0, 60, 1500)) + 0.005 * rng.standard_normal(1500)
    assert qc_ok(eda, HZ), qc_eda(eda, HZ)


def test_qc_rejects_rail_dropout():
    eda = np.full(1500, 0.030)  # electrode fell off
    assert not qc_ok(eda, HZ)
    assert qc_eda(eda, HZ)["rail_fraction"] == 1.0


def test_qc_rejects_all_nan():
    assert not qc_ok(np.full(1500, np.nan), HZ)


def test_qc_tolerates_quantization_repeats():
    # EmotiBit repeats values across LSL rows; short repeats are fine
    eda = np.repeat(np.linspace(0.1, 0.3, 300), 5)  # 80% exact-zero diffs
    assert qc_ok(eda, HZ), qc_eda(eda, HZ)


def test_qc_rejects_long_stuck_run():
    eda = np.concatenate([np.full(1200, 0.2), np.linspace(0.2, 0.3, 300)])
    q = qc_eda(eda, HZ)
    assert q["stuck_fraction"] > 0.5
    assert not qc_ok(eda, HZ)


def test_decompose_separates_timescales():
    t = np.arange(0, 120, 1 / HZ)
    slow = 0.05 * np.sin(2 * np.pi * 0.01 * t)          # tonic-ish drift
    fast = 0.02 * np.sin(2 * np.pi * 0.5 * t)           # phasic-ish wiggle
    out = decompose_eda(0.15 + slow + fast, HZ)
    # tonic should track the slow component, phasic the fast one
    assert np.corrcoef(out["eda_tonic"], slow)[0, 1] > 0.9
    assert np.corrcoef(out["eda_phasic"], fast)[0, 1] > 0.7
    # tonic + phasic reconstruct the original up to filter edge effects
    assert np.corrcoef(out["eda_tonic"] + out["eda_phasic"],
                       0.15 + slow + fast)[0, 1] > 0.99
