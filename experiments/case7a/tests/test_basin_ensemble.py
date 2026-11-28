"""Basin-ensemble manifest and the optional log-POD Case 7a hook."""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from basin_ensemble import (  # noqa: E402
    CASE7A_SOIL_M,
    CASE7A_VS_CONTACT,
    CASE7A_VS_ROCK,
    CASE7A_VS_SURFACE,
    CASE7A_XI,
    ZETA_RATIO,
    basin_rows,
    case7a_holdout_row,
    gradational_field,
    write_manifest,
)
from run_case7a import resolve_logpod  # noqa: E402


def test_train_rows_stay_inside_the_basin_ranges_and_holdout_is_separate():
    rows = basin_rows()
    train = [row for row in rows if row["split"] == "train"]
    holdout = [row for row in rows if row["holdout_case7a"] == 1]
    assert len(train) == 256
    assert len(holdout) == 1
    assert holdout[0]["split"] == "holdout"
    assert holdout[0]["sample_id"] == "holdout_case7a"
    assert holdout[0]["soil_thickness_m"] == CASE7A_SOIL_M
    assert holdout[0]["vs_surface_mps"] == CASE7A_VS_SURFACE
    assert holdout[0]["vs_contact_mps"] == CASE7A_VS_CONTACT
    assert holdout[0]["vs_rock_mps"] == CASE7A_VS_ROCK
    assert holdout[0]["rayleigh_xi"] == CASE7A_XI
    for row in train:
        assert row["holdout_case7a"] == 0
        assert 40.0 <= row["soil_thickness_m"] <= 120.0
        assert 100.0 <= row["vs_surface_mps"] <= 200.0
        assert 250.0 <= row["vs_contact_mps"] <= 600.0
        assert 800.0 <= row["vs_rock_mps"] <= 2600.0
        assert 0.005 <= row["rayleigh_xi"] <= 0.015
        assert row["zeta_ratio"] == ZETA_RATIO
        assert row["zeta_soil"] / row["zeta_rock"] == ZETA_RATIO
        assert row["downhole_depth_fraction"] == 0.8
        assert 20.0 <= row["contact_depth_west_m"] <= 140.0
        assert 20.0 <= row["contact_depth_east_m"] <= 140.0
    ids = [row["sample_id"] for row in rows]
    assert len(ids) == len(set(ids))


def test_gradational_field_puts_the_surface_on_row_zero():
    vs, rock = gradational_field(case7a_holdout_row())
    assert vs.shape == (150, 500)
    assert rock.shape == vs.shape
    assert vs[0].max() < 160.0
    assert np.all(vs[-1] == CASE7A_VS_ROCK)
    center = vs[:, 250]
    expected_surface = 140.0 + (0.5 / 85.0) * (353.0 - 140.0)
    assert np.isclose(center[0], expected_surface)
    assert np.count_nonzero(~rock[:, 250]) == 85
    dipped = {
        "contact_depth_west_m": 20.0,
        "contact_depth_east_m": 140.0,
        "vs_surface_mps": 100.0,
        "vs_contact_mps": 400.0,
        "vs_rock_mps": 2000.0,
    }
    _, dipped_rock = gradational_field(dipped)
    assert np.count_nonzero(~dipped_rock[:, 0]) < np.count_nonzero(~dipped_rock[:, -1])


def test_select_jobs_drops_the_holdout_and_a_finished_h5(tmp_path: Path, monkeypatch):
    import run_basin_ensemble as queue

    monkeypatch.setattr(queue, "_h5_ok", lambda path: path.name == "run_0.h5")
    jobs = queue.select_jobs(basin_rows(), tmp_path, limit=None, index=None, start=0)
    ids = [row["sample_id"] for _, row in jobs]
    assert "holdout_case7a" not in ids
    assert all(int(row["holdout_case7a"]) == 0 for _, row in jobs)
    assert [i for i, _ in jobs] == list(range(1, 256))


def test_choose_workers_caps_at_sixteen_and_falls_back_when_unmeasured():
    import run_basin_ensemble as queue

    assert queue.choose_workers(8.0, 222.0) == 16
    assert queue.choose_workers(20.0, 222.0) == 7
    assert queue.choose_workers(200.0, 222.0) == 1
    assert queue.choose_workers(0.0, 222.0) == 8


def test_manifest_roundtrip(tmp_path: Path):
    path = write_manifest(tmp_path / "basin_ensemble.csv", n_train=4, seed=1)
    text = path.read_text()
    assert "holdout_case7a" in text
    assert "Konno-Ohmachi 500" in text
    assert text.count("\n") == 6


def test_resolve_logpod_requires_both_files(tmp_path: Path):
    assert resolve_logpod(Namespace()) is None
    missing = Namespace(
        logpod_checkpoint=tmp_path / "missing.pt",
        logpod_model_dir=tmp_path,
    )
    assert resolve_logpod(missing) is None
    ckpt = tmp_path / "best_model.pt"
    ckpt.write_bytes(b"pt")
    np.save(tmp_path / "pod_modes.npy", np.zeros((1, 1, 2)))
    np.save(tmp_path / "pod_mean.npy", np.zeros((1, 2)))
    (tmp_path / "pod_space.txt").write_text("linear\n")
    ready = Namespace(logpod_checkpoint=ckpt, logpod_model_dir=tmp_path)
    try:
        resolve_logpod(ready)
    except ValueError as exc:
        assert "linear" in str(exc)
    else:
        raise AssertionError("linear basis was accepted")
    (tmp_path / "pod_space.txt").write_text("log\n")
    found = resolve_logpod(ready)
    assert found is not None
    assert found[0] == ckpt
    assert found[1] == tmp_path
