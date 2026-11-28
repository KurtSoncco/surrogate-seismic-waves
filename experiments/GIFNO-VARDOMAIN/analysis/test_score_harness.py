"""Metric definitions, aggregation refusal, provenance, and the 1e-6 identity."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "paper" / "scripts"))

import gates  # noqa: E402
import score_harness as harness  # noqa: E402
from metrics import FLOOR_LOG_L1, log_l1, pearson  # noqa: E402


def _curves(n: int = 4, n_freq: int = 32, seed: int = 0):
    rng = np.random.default_rng(seed)
    freq = np.linspace(0.1, 10.0, n_freq)
    log_true = rng.normal(size=(n, n_freq))
    log_prior = log_true + rng.normal(scale=0.2, size=(n, n_freq))
    log_pred = log_true + rng.normal(scale=0.05, size=(n, n_freq))
    return freq, log_pred, log_true, log_prior


def test_g_log_l1_and_overall_band_use_metrics():
    freq, log_pred, log_true, log_prior = _curves()
    g_pred = log_pred - log_prior
    g_true = log_true - log_prior
    for band in ("low", "mid", "high"):
        assert harness.g_log_l1(g_pred[0], g_true[0], freq, band) == pytest.approx(
            log_l1(g_pred[0], g_true[0], freq, band)
        )
    mask = harness.band_mask(freq, "overall")
    assert mask[0] and mask[-1]
    assert not harness.band_mask(np.array([0.099, 10.001]), "overall").any()
    assert harness.g_log_l1(g_pred[0], g_true[0], freq, "overall") == pytest.approx(
        log_l1(g_pred[0][mask], g_true[0][mask])
    )
    edges = np.array([0.1, 0.4, 2.5, 10.0])
    assert harness.band_mask(edges, "low").tolist() == [True, True, False, False]
    assert harness.band_mask(edges, "mid").tolist() == [False, True, True, False]
    assert harness.band_mask(edges, "high").tolist() == [False, False, True, True]


def test_skill_is_suppressed_at_the_gates_floor_and_not_at_metrics_floor():
    assert gates.NOISE_FLOOR_PLACEHOLDER_LOG_L1 != FLOOR_LOG_L1
    floor = gates.noise_floor_log_l1()
    assert harness.skill_value(0.01, floor) is None
    assert harness.skill_value(0.01, floor - 1e-6) is None
    above = harness.skill_value(0.01, floor + 1e-4)
    assert above == pytest.approx(1.0 - 0.01 / (floor + 1e-4))
    between = 0.5 * (gates.NOISE_FLOOR_PLACEHOLDER_LOG_L1 + FLOOR_LOG_L1)
    assert between < FLOOR_LOG_L1
    assert harness.skill_value(0.0, between) is not None


def test_pearson_is_log_tf_only():
    freq = np.linspace(0.1, 10.0, 40)
    log_true = np.linspace(-0.4, 0.8, freq.size)
    log_pred = log_true + 0.15 * np.sin(np.linspace(0.0, 6.0, freq.size))
    got = harness.pearson_log_tf(log_pred, log_true, freq, "overall")
    assert got == pytest.approx(pearson(log_pred, log_true))
    assert got != pytest.approx(pearson(np.exp(log_pred), np.exp(log_true)))


def test_harm_rate_is_the_share_of_records():
    freq = np.linspace(0.4, 2.5, 8)
    # Record a: both stations worse than g=0. Record b: both stations better.
    # A station-wise share would still be 0.5, so give record a one better station
    # whose pool is still worse, and record b one worse station whose pool is better.
    log_true = np.zeros((4, freq.size))
    log_prior = np.zeros((4, freq.size))
    log_pred = np.zeros((4, freq.size))
    log_prior[0] = 0.2
    log_prior[1] = 0.2
    log_prior[2] = 0.5
    log_prior[3] = 0.5
    log_pred[0] = 0.4
    log_pred[1] = 0.3
    log_pred[2] = 0.1
    log_pred[3] = 0.7
    table = harness.score_curves(
        sample_id=np.array(["a", "a", "b", "b"]),
        station=np.array([0, 1, 0, 1]),
        cluster_id=np.array(["m1", "m1", "m2", "m2"]),
        freq=freq,
        log_pred=log_pred,
        log_true=log_true,
        log_prior=log_prior,
        method="model",
    )
    # Record a pooled |pred| exceeds pooled |prior|; record b does not.
    assert harness.harm_rate(table, "overall", "model") == pytest.approx(0.5)
    flags = table.loc[(table["metric"] == "harmed") & (table["band"] == "overall"), "value"]
    assert float(np.mean(flags.to_numpy() > 0)) == pytest.approx(0.75)


def test_aggregation_refusal_keeps_the_two_skills_apart():
    freq, log_pred, log_true, log_prior = _curves(n=2, n_freq=16, seed=1)
    # Force the two reductions apart: one tiny prior and one large prior.
    log_prior = log_true.copy()
    log_prior[0] = log_true[0] + 0.05
    log_prior[1] = log_true[1] + 0.8
    log_pred = log_true.copy()
    log_pred[0] = log_true[0] + 0.04
    log_pred[1] = log_true[1] + 0.1
    table = harness.score_curves(
        sample_id=np.array(["r", "r"]),
        station=np.array([0, 1]),
        cluster_id=np.array(["g", "g"]),
        freq=freq,
        log_pred=log_pred,
        log_true=log_true,
        log_prior=log_prior,
        method="model",
    )
    pooled = harness.aggregate(table, "record")
    averaged = harness.aggregate_mean_skills(table, "record")
    band = "overall"
    pooled_skill = float(pooled.frame.loc[pooled.frame["band"] == band, "skill"].iloc[0])
    mean_skill = float(averaged.frame.loc[averaged.frame["band"] == band, "skill"].iloc[0])
    assert pooled_skill != pytest.approx(mean_skill)
    with pytest.raises(harness.AggregationError):
        harness.assert_comparable(pooled, averaged)
    same = harness.aggregate(table, "record")
    harness.assert_comparable(pooled, same)
    with pytest.raises(harness.AggregationError):
        harness.assert_comparable(pooled, harness.aggregate(table, "master"))


def test_bootstrap_refuses_stations_and_frequencies():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    records = np.array(["a", "a", "b", "b"])
    with pytest.raises(harness.AggregationError):
        harness.bootstrap_independent(values, np.arange(4), unit="station", n_draw=20, rng=np.random.default_rng(0))
    with pytest.raises(harness.AggregationError):
        harness.bootstrap_independent(values, np.arange(values.size), unit="frequency", n_draw=20, rng=np.random.default_rng(0))
    out = harness.bootstrap_independent(values, records, unit="record", n_draw=30, rng=np.random.default_rng(0))
    assert out["n_records"] == 2
    assert out["unit"] == "record"


def test_provenance_requires_the_phase0_fields(tmp_path: Path):
    path = tmp_path / "table.provenance.json"
    with pytest.raises(ValueError):
        harness.write_provenance(path, {"label_campaign": "iid"})
    fields = {key: "set" for key in harness.REQUIRED_PROVENANCE}
    fields["fft_window_s"] = None
    fields["epoch"] = 46
    fields["checkpoint"] = None
    harness.write_provenance(path, fields)
    saved = __import__("json").loads(path.read_text())
    for key in harness.REQUIRED_PROVENANCE:
        assert key in saved
    assert saved["sample_id_hash"] == "set"


def test_fixture_agrees_within_1e_6():
    freq, log_pred, log_true, log_prior = _curves(n=6, n_freq=24, seed=2)
    ids = np.array([f"s{i}" for i in range(6)])
    kwargs = dict(
        sample_id=ids,
        station=np.zeros(6, dtype=int),
        cluster_id=np.array(["g0", "g0", "g1", "g1", "g2", "g2"]),
        freq=freq,
        log_pred=log_pred,
        log_true=log_true,
        log_prior=log_prior,
        method="model",
    )
    first = harness.score_curves(**kwargs)
    second = harness.score_curves(**kwargs)
    left = first.loc[first["metric"] == "g_log_l1", "value"].to_numpy()
    right = second.loc[second["metric"] == "g_log_l1", "value"].to_numpy()
    assert np.max(np.abs(left - right)) <= 1e-6
    headline = harness.aggregate(first, "record")
    direct = []
    for i in range(6):
        direct.append(log_l1(log_pred[i] - log_prior[i], log_true[i] - log_prior[i], freq, "mid"))
    got = float(np.median(headline.frame.loc[headline.frame["band"] == "mid", "g_log_l1"]))
    assert abs(got - float(np.median(direct))) <= 1e-6
    digest = harness.sample_id_hash(ids)
    again = harness.sample_id_hash(list(reversed(ids)))
    assert digest == again
    assert len(digest) == 64
