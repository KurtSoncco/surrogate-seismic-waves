"""Stop and accept rules, and the sealed blind file. No training run."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gates  # noqa: E402
import train_monitor as mon  # noqa: E402


def test_stop_needs_three_epochs_or_harm_above_five_percent():
    quiet = {"iid": gates.HARM_RATE_STOP}
    two = [{"mirror": False}, {"mirror": False}]
    three = two + [{"mirror": False}]
    assert mon.consistency_fail_streak(two) == 2
    assert mon.stop_decision(two, quiet)["stop"] is False
    stopped = mon.stop_decision(three, quiet)
    assert stopped["stop"] is True
    assert "consistency" in stopped["reason"]
    mixed = [{"mirror": False}, {"cov0": False}, {"homogeneous_column": False}]
    assert mon.consistency_fail_streak(mixed) == 1
    assert mon.stop_decision(mixed, {"iid": 0.0, "manifold": gates.HARM_RATE_STOP})["stop"] is False
    harm = mon.stop_decision([{"mirror": True}], {"manifold": gates.HARM_RATE_STOP + 1e-6})
    assert harm["stop"] is True
    assert "manifold" in harm["reason"]
    assert mon.stop_decision([{"homogeneous_column": True}], {"iid": 0.0})["stop"] is False


def test_accept_rule_against_p128_margin_half_width_and_seeds():
    base = dict(
        overall_val=0.04,
        baseline_overall=0.10,
        half_width=0.02,
        family_val={"iid": 0.05, "manifold": 0.06},
        family_baseline={"iid": 0.05, "manifold": 0.04},
        consistency_passed=True,
        n_seeds=gates.MIN_TRAINING_SEEDS,
    )
    assert base["family_val"]["manifold"] - base["family_baseline"]["manifold"] == pytest.approx(gates.TOST_DELTA_LOG_L1)
    assert mon.accept_versus_p128(**base)["accept"] is True
    tied = dict(base, overall_val=0.125, baseline_overall=0.25, half_width=0.125)
    assert mon.accept_versus_p128(**tied)["improvement"] == pytest.approx(0.125)
    assert mon.accept_versus_p128(**tied)["accept"] is False
    worse = dict(base, family_val={"iid": 0.05, "manifold": 0.04 + gates.TOST_DELTA_LOG_L1 + 0.001})
    assert mon.accept_versus_p128(**worse)["accept"] is False
    short = dict(base, n_seeds=gates.MIN_TRAINING_SEEDS - 1)
    refused = mon.accept_versus_p128(**short)
    assert refused["accept"] is False
    assert "seeds" in " ".join(refused["blocked_by"])
    failed = dict(base, consistency_passed=False)
    assert mon.accept_versus_p128(**failed)["accept"] is False
    assert mon.consistency_passed_for_accept({"homogeneous_column": True, "cov0": True}) is False
    assert mon.consistency_passed_for_accept({name: True for name in mon.ACCEPT_PROBES}) is True


def test_reader_refuses_before_opening_and_the_open_log_omits_blind(tmp_path):
    missing = tmp_path / "absent.csv"
    with pytest.raises(mon.BlindSealed) as sealed:
        mon.read_sealed_blind(missing)
    assert "blind-0" not in str(sealed.value)
    with pytest.raises(FileNotFoundError):
        mon.read_sealed_blind(missing, open_blind=True)

    table = mon.synthetic_table()
    floors = mon.Floors(pod={("iid", "overall"): 0.01, ("manifold", "overall"): 0.02}, realization={("iid", "overall"): 0.05})
    frame = mon.log_epoch(
        table,
        epoch=0,
        out_dir=tmp_path,
        floors=floors,
        consistency={"homogeneous_column": True, "cov0": True, "mirror": True},
        baseline_overall=1.0,
        baseline_by_family={"iid": 1.0, "manifold": 1.0},
        n_seeds=1,
        n_draw=20,
        rng=np.random.default_rng(1),
    )
    assert list(frame.columns) == list(mon.MONITOR_COLUMNS)
    text = (tmp_path / mon.MONITOR_NAME).read_text()
    assert "blind-0" not in text
    assert set(frame["family"]) == {"iid", "manifold"}
    assert set(frame["band"]) == {"low", "mid", "high", "overall"}
    assert set(frame["accept"]) == {"no"}
    with pytest.raises(mon.BlindSealed) as closed:
        mon.read_sealed_blind(tmp_path / mon.SEALED_NAME)
    assert "blind-0" not in str(closed.value)
    opened = mon.read_sealed_blind(tmp_path / mon.SEALED_NAME, open_blind=True)
    assert "blind-0" in set(opened["sample_id"].astype(str))
    assert opened.loc[opened["sample_id"].astype(str) == "blind-0", "metric"].eq("g_log_l1").any()


def test_audit_floors_match_the_csv_median_and_pod_column():
    floors = mon.load_audit_floors(HERE / "audit", split="val")
    rep = pd.read_csv(HERE / "audit" / "representation.csv")
    pod = rep.loc[(rep["table"] == "pod_floor") & (rep["pool"] == "iid:val") & (rep["band"] == "overall"), "pod_floor_l1"]
    assert floors.pod_at("iid", "overall") == pytest.approx(float(pod.iloc[0]))
    assert floors.pod_at("manifold", "overall") > 0.0
    real = pd.read_csv(HERE / "audit" / "realization.csv")
    med = real.loc[(real["family"] == "iid") & (real["band"] == "overall"), "value"].median()
    assert floors.realization_at("iid", "overall") == pytest.approx(float(med))
    assert np.isnan(floors.realization_at("manifold", "overall"))


def test_dry_run_writes_the_monitor_columns(tmp_path):
    frame = mon.dry_run(tmp_path, HERE / "audit")
    assert list(frame.columns) == list(mon.MONITOR_COLUMNS)
    assert (frame["stop"] == "no").all()
    assert (frame["accept"] == "no").all()
    assert frame["accept_blocked"].str.contains("seeds").all()
    assert frame.loc[frame["family"] == "iid", "gap_pod"].notna().all()
    assert frame.loc[frame["family"] == "manifold", "realization_floor"].isna().all()
