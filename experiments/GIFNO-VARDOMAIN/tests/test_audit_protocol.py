"""Preregistered nesting, representation, diagnosis, and the sealed external audit."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import choose_groups, choose_iid_flats, nested_prefix, training_records  # noqa: E402
from diagnosis import classify_learning_curve, length_scaling_gate, projected_gain, fit_power_law  # noqa: E402
from external.audit import confirmatory_decision, family_gate, geology_interval  # noqa: E402
from external.decision import paper_decision  # noqa: E402
from external.eligibility import classify_section  # noqa: E402
from external.seal import Ledger, open_bytes, seal_bytes  # noqa: E402
from external.sites import SITES  # noqa: E402
from external.windows import confirmatory_stations  # noqa: E402
from model import CoupledPODMap, VarDomainLOGLO  # noqa: E402
from representation import fit_input_pod, reduced_rank_map, shuffled_pair_gap  # noqa: E402
from train import load_frozen_basis  # noqa: E402


def test_iid_and_width_prefixes_are_nested():
    grouped = {group: list(range(group * 10, group * 10 + 8)) for group in range(20)}
    small = choose_iid_flats(grouped, 0.25, 2, 0)
    middle = choose_iid_flats(grouped, 0.50, 4, 0)
    full = choose_iid_flats(grouped, 1.0, None, 0)
    assert small < middle < full
    groups = list(range(12))
    widths = {group: (500, 1000, 1500, 2000)[group // 3] for group in groups}
    first = set(nested_prefix(groups, 0.25, 0, widths=widths))
    second = set(nested_prefix(groups, 0.50, 0, widths=widths))
    assert first <= second
    assert {widths[group] for group in first} == {500, 1000, 1500, 2000}
    assert choose_groups(groups, 0.25, 0, widths) <= choose_groups(groups, 1.0, 0, widths)


def test_blocked_splits_and_external_bins_cannot_train():
    with pytest.raises(ValueError):
        training_records(None, 0, "external_blind")
    with pytest.raises(ValueError):
        training_records(None, 0, "test")


def test_weighted_pod_and_coupled_map_do_not_read_the_target_at_inference():
    rng = np.random.default_rng(0)
    vs = np.exp(rng.normal(size=(80, 12)))
    zeta = np.full_like(vs, 0.02)
    observed = np.ones(vs.shape, dtype=bool)
    basis = fit_input_pod(vs, zeta, observed, 8)
    assert basis["components"].shape[0] <= 8
    codes = rng.normal(size=(100, 4))
    truth = codes @ np.array([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5], [0.2, 0.2]])
    gap = shuffled_pair_gap(codes, truth, 2)
    assert gap["passes"]
    mapping = reduced_rank_map(codes, truth, 2)
    module = CoupledPODMap(mapping["beta"], mapping["x_mean"], mapping["y_mean"])
    assert list(inspect.signature(CoupledPODMap.forward).parameters) == ["self", "code"]
    assert list(inspect.signature(VarDomainLOGLO.forward).parameters) == ["self", "x", "query", "n_keep"]


def test_learning_curve_rules_and_length_scaling():
    size = np.array([200.0, 400.0, 800.0, 1600.0])
    falling = 0.05 + 20.0 * size ** -0.5
    result = classify_learning_curve(size, falling, e_floor=0.01, e_1d=1.0, n_boot=80, seed=1)
    assert result["label"] == "still_falling"
    assert projected_gain(fit_power_law(size, falling), 1600.0) > 0.02
    flat = np.full(4, 0.20)
    near = classify_learning_curve(size, flat, e_floor=0.19, e_1d=1.0, n_boot=40, seed=2)
    assert near["label"] in {"flat_near_floor", "inconclusive"}
    assert length_scaling_gate(np.array([0.01, 0.01, 0.02]), 0.0)["passes"] is False
    assert length_scaling_gate(np.array([0.001, 0.002, 0.003]), 0.0)["passes"] is True
    assert length_scaling_gate(np.array([]), 0.0)["passes"] is False


def test_three_sites_are_not_a_confirmatory_cohort():
    decision = paper_decision(timing_measured=False, custodian="", cuda=False, sites=SITES)
    assert decision["paper"] == "paper_3_descriptive"
    assert decision["rebuild_paper_2"] is False
    assert decision["reported_seed"] == 0
    assert decision["reported_model"] == "not_trained"
    interval = geology_interval({"a": 0.1, "b": 0.2, "c": 0.3})
    assert interval["confirmatory"] is False
    families = family_gate({"flat": {"a": 0.1}, "basin": {"a": 0.2, "b": 0.3, "c": 0.4}})
    assert families["inferential"] == ["basin"]
    primary = confirmatory_decision(
        model_error={"a": 0.2}, column_error={"a": 0.1}, reference_error={"a": 0.05},
        adequacy=0.3, margin=0.02, noninferiority=0.02,
    )
    assert primary["primary_pass"] is False
    assert primary["secondary_tested"] is False


def test_eligibility_window_and_seal():
    envelope = {"vs_min": 100.0, "vs_max": 2000.0, "impedance_ratio_max": 10.0}
    inside = classify_section({
        "width_m": 1000, "depth_m": 80, "flat_surface": True, "elastic_half_space": True,
        "vs_min": 150, "vs_max": 800,
    }, envelope)
    outside = classify_section({
        "width_m": 4000, "depth_m": 600, "flat_surface": False, "elastic_half_space": False,
        "vs_min": 50, "vs_max": 3000,
    }, envelope)
    assert inside["confirmatory"] is True
    assert outside["tier"] == "extrapolation"
    short = confirmatory_stations(800)
    assert short["confirmatory_x_m"] == []
    long = confirmatory_stations(3000)
    assert long["confirmatory_x_m"]
    assert min(long["confirmatory_x_m"]) >= 1000
    sealed = seal_bytes(b"labels", b"custodian-key")
    with pytest.raises(PermissionError):
        open_bytes(sealed, b"custodian-key", ledger_opened=False)
    assert open_bytes(sealed, b"custodian-key", ledger_opened=True) == b"labels"
    ledger = Ledger(Path(pytest.__file__).with_name("_not_used"))
    path = Path(__file__).with_name("_ledger_test.jsonl")
    path.unlink(missing_ok=True)
    ledger = Ledger(path)
    ledger.append({"event": "open", "payload_id": "site-a"})
    with pytest.raises(PermissionError):
        ledger.append({"event": "open", "payload_id": "site-a"})
    path.unlink()


def test_frozen_basis_does_not_return_network_weights(tmp_path: Path):
    rng = np.random.default_rng(0)
    modes = rng.normal(size=(4, 6)).astype(np.float32)
    mean = np.zeros(6, np.float32)
    np.save(tmp_path / "pod_modes.npy", modes)
    np.save(tmp_path / "pod_mean.npy", mean)
    torch.save({"state_dict": {"pod_coeff_scale": torch.ones(4), "secret_weight": torch.zeros(3)}}, tmp_path / "best_model.pt")
    loaded = load_frozen_basis(tmp_path)
    assert set(loaded) == {"modes", "mean", "weight", "scale", "pca"}
    assert "secret_weight" not in loaded
    np.testing.assert_allclose(loaded["scale"], np.ones(4, np.float32))
