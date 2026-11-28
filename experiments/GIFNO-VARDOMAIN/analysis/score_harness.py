#!/usr/bin/env python3
"""One scorer for the variable-domain revision.

Bands, log-L1, and Pearson come from ``paper/scripts/metrics.py``. Cluster
resampling comes from ``paper/scripts/bootstrap_ci.py``. Skill is suppressed
at the placeholder in ``gates.py``. ``metrics.FLOOR_LOG_L1`` is not read.

The long table has one row per ``(sample_id, station, band, metric)``.
Headline skill is the skill of the pooled absolute ``g`` log-L1 at an
explicit unit: ``record``, ``station``, or ``master``. The mean of per-unit
skills is a different reduction. ``assert_comparable`` refuses to put the
two in one comparison.

    python score_harness.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_ANALYSIS = Path(__file__).resolve().parent
_SCRIPTS = _ANALYSIS.parent / "paper" / "scripts"
for _path in (_ANALYSIS, _SCRIPTS):
    text = str(_path)
    if text not in sys.path:
        sys.path.insert(0, text)

from bootstrap_ci import block_bootstrap_median, bootstrap_statistic  # noqa: E402
from metrics import log_l1 as metrics_log_l1  # noqa: E402
from metrics import pearson as metrics_pearson  # noqa: E402
from metrics import skill as metrics_skill  # noqa: E402
from metrics import band_mask as metrics_band_mask  # noqa: E402

from gates import BOOTSTRAP_ALPHA, noise_floor_log_l1  # noqa: E402

ANALYSIS = Path(__file__).resolve().parent
VARDOMAIN = ANALYSIS.parent
REPO = VARDOMAIN.parents[1]
CACHE = REPO / "data" / "vardomain" / "cache"

BAND_NAMES = ("low", "mid", "high", "overall")
LONG_COLUMNS = (
    "sample_id",
    "station",
    "cluster_id",
    "x_m",
    "method",
    "band",
    "metric",
    "value",
)
METRIC_NAMES = ("g_log_l1", "prior_log_l1", "skill", "pearson_log_tf", "harmed")
AGGREGATION_UNITS = ("record", "station", "master")
SKILL_POOLED = "pooled"
SKILL_MEAN = "mean_of_unit_skills"
AGREE_ABS = 1e-6

REQUIRED_PROVENANCE = (
    "label_campaign",
    "oned_engine",
    "outcrop_convention",
    "damping_source",
    "fft_window_s",
    "checkpoint",
    "epoch",
    "dtype",
    "bands",
    "aggregation_unit",
    "sample_id_hash",
    "git_commit",
)
# Classical arms have no checkpoint. A campaign whose FFT window is not in
# ``cache/meta.json`` leaves ``fft_window_s`` null and says so in the sidecar.
NONE_ALLOWED = ("checkpoint", "epoch", "fft_window_s")


class AggregationError(ValueError):
    """A comparison mixed units or mixed skill reductions."""


def band_mask(freq: np.ndarray, name: str) -> np.ndarray:
    """Inclusive band. ``overall`` is the union of the three ``metrics`` bands."""

    if name == "overall":
        mask = metrics_band_mask(freq, "low")
        mask = mask | metrics_band_mask(freq, "mid")
        mask = mask | metrics_band_mask(freq, "high")
        return mask
    if name not in ("low", "mid", "high"):
        raise KeyError(name)
    return metrics_band_mask(freq, name)


def g_log_l1(g_pred: np.ndarray, g_true: np.ndarray, freq: np.ndarray, band: str) -> float:
    """Mean ``|g_pred − g_true|`` on ``band``. This is ``metrics.log_l1``."""

    pred = np.asarray(g_pred, dtype=np.float64)
    true = np.asarray(g_true, dtype=np.float64)
    if band == "overall":
        mask = band_mask(freq, "overall")
        if not np.any(mask):
            raise ValueError("overall band is empty")
        return metrics_log_l1(pred[..., mask], true[..., mask])
    return metrics_log_l1(pred, true, freq, band)


def weighted_overall_g_log_l1(g_pred: np.ndarray, g_true: np.ndarray, freq: np.ndarray):
    """Absolute ``g`` log-L1 on the harness overall band, weighted by ``sqrt(f / 1 Hz)``.

    The weight is ``model.spectral_weight``, the POD basis weight. Frequencies
    outside 0.1–10 Hz are left out. This is not the unweighted ``g_log_l1``,
    and it is not the training loss: that loss also doubles 0.1–0.4 Hz on
    windows of 1000 m and wider.

    A 1-D curve returns a float. A batch whose last axis is frequency returns
    one value per leading index. Passing ``log|TF|`` pairs that share a prior
    is the same difference as passing ``g``.
    """

    _vardomain_on_path()
    from model import spectral_weight

    pred = np.asarray(g_pred, dtype=np.float64)
    true = np.asarray(g_true, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError("g_pred and g_true must share a shape")
    freq = np.asarray(freq, dtype=np.float64)
    if freq.ndim != 1 or pred.shape[-1] != freq.shape[0]:
        raise ValueError("the last axis must be frequency")
    mask = band_mask(freq, "overall")
    if not np.any(mask):
        raise ValueError("overall band is empty")
    err = np.abs(pred[..., mask] - true[..., mask])
    weight = np.asarray(spectral_weight(freq)[mask], dtype=np.float64)
    reduced = np.sum(err * weight, axis=-1) / np.sum(weight)
    if reduced.ndim == 0:
        return float(reduced)
    return reduced


def prior_log_l1(g_true: np.ndarray, freq: np.ndarray, band: str) -> float:
    """``L1(g_true)``, the absolute error of the correction ``g = 0``."""

    true = np.asarray(g_true, dtype=np.float64)
    return g_log_l1(np.zeros_like(true), true, freq, band)


def skill_value(model_l1: float, prior_l1: float, floor: float | None = None) -> float | None:
    """``1 − L1(g_pred − g_true) / L1(g_true)`` when the prior exceeds the floor.

    At or below the floor the skill is suppressed. Above the floor the value
    is ``metrics.skill``. The ``1e-6`` inside that function is inactive
    because the floor is larger.
    """

    floor = noise_floor_log_l1() if floor is None else float(floor)
    prior = float(prior_l1)
    if not np.isfinite(prior) or prior <= floor:
        return None
    return metrics_skill(float(model_l1), prior)


def pearson_log_tf(log_a: np.ndarray, log_b: np.ndarray, freq: np.ndarray, band: str) -> float:
    """Pearson of ``log|TF|`` on ``band``. Linear ``|TF|`` is not accepted."""

    mask = band_mask(freq, band)
    if not np.any(mask):
        raise ValueError(f"{band} band is empty")
    return metrics_pearson(np.asarray(log_a, dtype=np.float64)[..., mask], np.asarray(log_b, dtype=np.float64)[..., mask])


def skill_of_pooled(model_l1: np.ndarray, prior_l1: np.ndarray, floor: float | None = None) -> float | None:
    """Skill of the mean absolute error. Equal-length stations pool by a mean."""

    return skill_value(float(np.mean(model_l1)), float(np.mean(prior_l1)), floor)


def mean_of_unit_skills(skills: np.ndarray) -> float:
    """Mean of per-unit skills. Do not compare this with ``skill_of_pooled``."""

    values = np.asarray(skills, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


@dataclass(frozen=True)
class Headline:
    frame: pd.DataFrame
    aggregation_unit: str
    skill_reduction: str

    def as_dict(self) -> dict:
        return {"aggregation_unit": self.aggregation_unit, "skill_reduction": self.skill_reduction}


def assert_comparable(left: Headline, right: Headline) -> None:
    """Refuse a comparison that mixes units or mixes the two skill reductions."""

    if left.aggregation_unit != right.aggregation_unit or left.skill_reduction != right.skill_reduction:
        raise AggregationError(
            "refusing to mix "
            f"{left.skill_reduction} on {left.aggregation_unit} with "
            f"{right.skill_reduction} on {right.aggregation_unit}"
        )


def score_curves(
    *,
    sample_id: np.ndarray,
    station: np.ndarray,
    cluster_id: np.ndarray,
    freq: np.ndarray,
    log_pred: np.ndarray,
    log_true: np.ndarray,
    log_prior: np.ndarray,
    method: str,
    x_m: np.ndarray | None = None,
    floor: float | None = None,
) -> pd.DataFrame:
    """Long table. ``g = log|TF| − log|TF_1D|`` and both logs use the same prior."""

    pred = np.asarray(log_pred, dtype=np.float64)
    true = np.asarray(log_true, dtype=np.float64)
    prior = np.asarray(log_prior, dtype=np.float64)
    sample_id = np.asarray(sample_id)
    station = np.asarray(station)
    cluster_id = np.asarray(cluster_id)
    freq = np.asarray(freq, dtype=np.float64)
    if pred.ndim != 2 or pred.shape != true.shape or pred.shape != prior.shape:
        raise ValueError("log_pred, log_true, and log_prior must share shape (n, n_freq)")
    n, n_freq = pred.shape
    if freq.shape != (n_freq,):
        raise ValueError("freq must have one entry per frequency column")
    if sample_id.shape != (n,) or station.shape != (n,) or cluster_id.shape != (n,):
        raise ValueError("sample_id, station, and cluster_id must have one entry per curve")
    if x_m is None:
        x_values = np.full(n, np.nan)
    else:
        x_values = np.asarray(x_m, dtype=np.float64)
        if x_values.shape != (n,):
            raise ValueError("x_m must have one entry per curve")
    floor = noise_floor_log_l1() if floor is None else float(floor)
    g_pred = pred - prior
    g_true = true - prior
    rows: list[dict] = []
    for i in range(n):
        for band in BAND_NAMES:
            model = g_log_l1(g_pred[i], g_true[i], freq, band)
            base = prior_log_l1(g_true[i], freq, band)
            skill = skill_value(model, base, floor)
            pear = pearson_log_tf(pred[i], true[i], freq, band)
            harmed = float(model > base)
            values = {
                "g_log_l1": model,
                "prior_log_l1": base,
                "skill": np.nan if skill is None else skill,
                "pearson_log_tf": pear,
                "harmed": harmed,
            }
            for metric, value in values.items():
                rows.append({
                    "sample_id": sample_id[i],
                    "station": station[i],
                    "cluster_id": cluster_id[i],
                    "x_m": float(x_values[i]),
                    "method": method,
                    "band": band,
                    "metric": metric,
                    "value": value,
                })
    frame = pd.DataFrame(rows, columns=list(LONG_COLUMNS))
    return frame


def _subset(table: pd.DataFrame, metric: str) -> pd.DataFrame:
    return table.loc[table["metric"] == metric, ["method", "sample_id", "station", "cluster_id", "band", "value"]]


def aggregate(table: pd.DataFrame, unit: str) -> Headline:
    """Pool absolute ``g`` log-L1 first, then take the skill of that pool."""

    if unit not in AGGREGATION_UNITS:
        raise AggregationError(f"aggregation unit must be one of {AGGREGATION_UNITS}, got {unit!r}")
    if unit == "record":
        keys = ["method", "band", "sample_id"]
    elif unit == "station":
        keys = ["method", "band", "sample_id", "station"]
    else:
        keys = ["method", "band", "cluster_id"]
    identity = ["method", "sample_id", "station", "cluster_id", "band"]
    model = _subset(table, "g_log_l1").rename(columns={"value": "g_log_l1"})
    base = _subset(table, "prior_log_l1").rename(columns={"value": "prior_log_l1"})
    pear = _subset(table, "pearson_log_tf").rename(columns={"value": "pearson_log_tf"})
    merged = model.merge(base, on=identity, how="inner").merge(pear, on=identity, how="inner")
    # Equal frequency counts make the mean of station log-L1 the pooled mean.
    grouped = merged.groupby(keys, sort=False)
    rows = []
    for key, part in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        spec = dict(zip(keys, key))
        model_l1 = float(part["g_log_l1"].mean())
        prior_l1 = float(part["prior_log_l1"].mean())
        skill = skill_of_pooled(part["g_log_l1"].to_numpy(), part["prior_log_l1"].to_numpy())
        rows.append({
            **spec,
            "n_stations": int(len(part)),
            "g_log_l1": model_l1,
            "prior_log_l1": prior_l1,
            "skill": np.nan if skill is None else skill,
            "pearson_log_tf": float(part["pearson_log_tf"].median()),
            "harmed": float(model_l1 > prior_l1),
        })
    frame = pd.DataFrame(rows)
    frame.attrs["aggregation_unit"] = unit
    frame.attrs["skill_reduction"] = SKILL_POOLED
    return Headline(frame, unit, SKILL_POOLED)


def aggregate_mean_skills(table: pd.DataFrame, unit: str) -> Headline:
    """Mean of per-station skills inside the unit. Not the pooled skill."""

    if unit not in AGGREGATION_UNITS:
        raise AggregationError(f"aggregation unit must be one of {AGGREGATION_UNITS}, got {unit!r}")
    if unit == "record":
        keys = ["method", "band", "sample_id"]
    elif unit == "station":
        keys = ["method", "band", "sample_id", "station"]
    else:
        keys = ["method", "band", "cluster_id"]
    skills = _subset(table, "skill")
    rows = []
    for key, part in skills.groupby(keys, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        spec = dict(zip(keys, key))
        rows.append({**spec, "n_stations": int(len(part)), "skill": mean_of_unit_skills(part["value"].to_numpy())})
    frame = pd.DataFrame(rows)
    return Headline(frame, unit, SKILL_MEAN)


def harm_rate(table: pd.DataFrame, band: str, method: str) -> float:
    """Share of records whose pooled log-L1 is higher than ``g = 0``.

    The record is ``sample_id``. Stations inside a record are pooled before
    the comparison. Ties are not harm.
    """

    part = table.loc[(table["band"] == band) & (table["method"] == method)]
    model = part.loc[part["metric"] == "g_log_l1"].groupby("sample_id")["value"].mean()
    base = part.loc[part["metric"] == "prior_log_l1"].groupby("sample_id")["value"].mean()
    model, base = model.align(base, join="inner")
    if model.empty:
        raise ValueError(f"no records for method {method!r} on band {band!r}")
    return float(np.mean(model.to_numpy() > base.to_numpy()))


def bootstrap_independent(
    values,
    unit_ids,
    *,
    unit: str,
    statistic=np.median,
    n_draw: int | None = None,
    rng: np.random.Generator | None = None,
    alpha: float = BOOTSTRAP_ALPHA,
) -> dict:
    """Resample ``unit``. Stations and frequencies are refused."""

    if unit in ("station", "frequency"):
        raise AggregationError("cluster bootstrap resamples the independent unit, never stations or frequencies")
    if unit not in ("record", "master"):
        raise AggregationError(f"bootstrap unit must be 'record' or 'master', got {unit!r}")
    kwargs = {"rng": rng, "alpha": alpha}
    if n_draw is not None:
        kwargs["n_draw"] = int(n_draw)
    out = bootstrap_statistic(values, unit_ids, statistic, **kwargs)
    out["unit"] = unit
    return out


def bootstrap_moving_block(
    values,
    x_m,
    *,
    n_draw: int | None = None,
    rng: np.random.Generator | None = None,
    alpha: float = BOOTSTRAP_ALPHA,
) -> dict:
    """Treasure Island: one section, moving block along x."""

    kwargs = {"rng": rng, "alpha": alpha}
    if n_draw is not None:
        kwargs["n_draw"] = int(n_draw)
    out = block_bootstrap_median(values, x_m, **kwargs)
    out["unit"] = "moving block along x"
    return out


def sample_id_hash(sample_ids) -> str:
    digest = hashlib.sha256()
    for item in sorted({str(value) for value in sample_ids}):
        digest.update(item.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def git_commit() -> str:
    done = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    if done.returncode != 0:
        return "unknown"
    return done.stdout.strip()


AUDIT_PROVENANCE = REQUIRED_PROVENANCE + (
    "preregistration_sha256",
    "external_manifest_sha256",
    "checkpoint_sha256",
    "geology_ids",
    "ledger_event",
)


def equal_geology_mean(values) -> float:
    """Mean of one number per independent geology. Stations are not the unit."""
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("geology mean needs at least one geology")
    return float(array.mean())


def write_audit_provenance(path: Path, fields: dict) -> None:
    """Provenance for an external audit. The extra hashes are mandatory."""
    missing = [key for key in AUDIT_PROVENANCE if key not in fields]
    if missing:
        raise ValueError(f"audit provenance is missing {missing}")
    if fields.get("ledger_event") != "open":
        raise ValueError("audit provenance requires the ledger opening event")
    write_provenance(path, fields)


def write_provenance(path: Path, fields: dict) -> None:
    """Write ``*.provenance.json``. Every required field is present."""

    missing = [key for key in REQUIRED_PROVENANCE if key not in fields]
    if missing:
        raise ValueError(f"provenance is missing {missing}")
    blank = [
        key for key in REQUIRED_PROVENANCE
        if key not in NONE_ALLOWED and fields[key] in (None, "")
    ]
    if blank:
        raise ValueError(f"provenance left {blank} empty")
    payload = dict(fields)
    payload.setdefault("definition_difference", "")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _vardomain_on_path() -> None:
    text = str(VARDOMAIN)
    if text not in sys.path:
        sys.path.insert(0, text)


def compare_dtype(model_dir: Path | None = None, out_dir: Path | None = None) -> dict:
    """One fixed window through ``load_fp32`` and ``evaluate.load_model``.

    No CUDA means training autocast stays off, so both calls are float32.
    The maximum absolute ``g`` log-L1 between those two forwards is recorded.
    Float32 stays the frozen dtype unless that difference exceeds the
    noise-floor placeholder. This is not a full-grid sweep.
    """

    import torch

    _vardomain_on_path()
    from data import IID_MESH_M, KIND_RUN, _load_campaign, make_window
    from diag_forward import load_fp32
    from evaluate import load_model
    from provenance import inference_fields

    model_dir = Path(model_dir or REPO / "checkpoints" / "vardomain" / "v2_ablations" / "P128n128")
    out_dir = Path(out_dir or ANALYSIS)
    camp = _load_campaign("iid", CACHE / "iid")
    split = np.asarray(camp.split).astype(str)
    valid_row = np.asarray(camp.valid).astype(bool)
    hits = np.flatnonzero((split == "test") & valid_row)
    if hits.size == 0:
        raise RuntimeError("IID cache has no valid test row")
    index = int(hits[0])
    width = 500
    start = int(IID_MESH_M // 2 - width // 2)

    class _OneCampaign:
        def campaign(self, flat: int):
            return camp, int(flat)

    x, query, _tf, _prior, valid, info, _x_abs = make_window(
        _OneCampaign(), "metric", KIND_RUN, index, width, start, 0, False,
    )
    freq = np.asarray(np.load(CACHE / "freq.npy"), dtype=np.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_fp32, stamp = load_fp32(model_dir, device)
    model_train, cfg = load_model(model_dir, device)
    # Both scoring paths call eval. load_model itself leaves the module as constructed.
    model_fp32.eval()
    model_train.eval()
    amp = bool(cfg.get("amp", True))
    autocast = bool(inference_fields(device, amp)["autocast"])
    weight_gap = 0.0
    left = model_fp32.state_dict()
    right = model_train.state_dict()
    for key in left:
        if key in right and torch.is_floating_point(left[key]):
            weight_gap = max(weight_gap, float((left[key] - right[key]).abs().max()))
    with torch.inference_mode():
        g_fp32 = model_fp32(x.unsqueeze(0).to(device), query.unsqueeze(0).to(device)).float().cpu().numpy()[0]
        g_train = model_train(x.unsqueeze(0).to(device), query.unsqueeze(0).to(device)).float().cpu().numpy()[0]
    keep = np.flatnonzero(np.asarray(valid))
    if keep.size == 0:
        raise RuntimeError("the fixed window has no labeled stations")
    per_band = {}
    worst = 0.0
    for band in BAND_NAMES:
        band_worst = 0.0
        for station in keep:
            diff = g_log_l1(g_fp32[int(station)], g_train[int(station)], freq, band)
            band_worst = max(band_worst, diff)
        per_band[band] = band_worst
        worst = max(worst, band_worst)
    floor = noise_floor_log_l1()
    frozen = "float32" if worst <= floor else "unfrozen"
    payload = {
        "scope": "one centered 500 m IID test window; not a full-grid dtype sweep",
        "model_dir": str(model_dir),
        "epoch": None if stamp.get("epoch") is None else int(stamp["epoch"]),
        "local_index": index,
        "group_id": int(camp.group[index]),
        "width_m": width,
        "start_m": start,
        "n_stations": int(keep.size),
        "device": device.type,
        "cuda": bool(torch.cuda.is_available()),
        "training_amp_config": amp,
        "autocast_engaged": autocast,
        "load_fp32": "diag_forward.load_fp32",
        "load_train": "evaluate.load_model",
        "max_abs_weight": weight_gap,
        "weight_keys_match": set(left) == set(right),
        "max_abs_g_log_l1": worst,
        "max_abs_g_log_l1_by_band": per_band,
        "noise_floor_placeholder": floor,
        "frozen_dtype": frozen,
        "decision": (
            "float32 stays frozen because the measured g log-L1 difference does not exceed the noise-floor placeholder"
            if frozen == "float32"
            else "float32 is not frozen: the measured g log-L1 difference exceeds the noise-floor placeholder"
        ),
    }
    if not autocast:
        payload["autocast_note"] = (
            "evaluate.load_model binds bfloat16 autocast only on CUDA. "
            "This process has no CUDA, so the training loader ran in float32. "
            "The recorded difference is that CPU pair. A GPU bf16 sweep was not run."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dtype_comparison.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_provenance(out_dir / "dtype_comparison.provenance.json", {
        "label_campaign": "iid",
        "oned_engine": "not used; the comparison is g against g",
        "outcrop_convention": "not used",
        "damping_source": "qvs",
        "fft_window_s": None,
        "checkpoint": str(model_dir),
        "epoch": payload["epoch"],
        "dtype": frozen,
        "bands": list(BAND_NAMES),
        "aggregation_unit": "station",
        "bootstrap_unit": "not resampled; one fixed window",
        "sample_id_hash": sample_id_hash([f"iid:{index}"]),
        "git_commit": git_commit(),
        "definition_difference": payload.get("autocast_note", ""),
        "max_abs_g_log_l1": worst,
        "noise_floor_placeholder": floor,
    })
    return payload


def _logs(curve: np.ndarray, clip: float) -> np.ndarray:
    return np.log(np.clip(np.asarray(curve, dtype=np.float64), clip, None))


def _stack(blob, key: str, order: dict[int, int], runs: np.ndarray) -> np.ndarray:
    return np.stack([np.asarray(blob[key][order[int(run)]], dtype=np.float64) for run in runs])


def _median_column(frame: pd.DataFrame, method: str, band: str, column: str) -> float:
    part = frame.loc[(frame["method"] == method) & (frame["band"] == band), column]
    if part.empty:
        raise KeyError(f"{method} {band} {column}")
    return float(np.median(part.to_numpy()))


def _published_practice(table: pd.DataFrame, domain: str, method: str, physics: str) -> pd.Series:
    part = table[
        (table["domain"] == domain)
        & (table["method"] == method)
        & (table["physics"] == physics)
    ]
    if len(part) != 1:
        raise KeyError(f"{domain} {method} {physics}")
    return part.iloc[0]


def _score_family_curves(family: str, runs, group, freq, truth, column, curves: dict[str, np.ndarray], clip: float) -> pd.DataFrame:
    log_true = _logs(truth, clip)
    log_prior = _logs(column, clip)
    sample = np.array([f"{family}:{int(run)}" for run in runs])
    station = np.full(runs.shape[0], 10)
    cluster = np.asarray(group).astype(str)
    frames = []
    for method, curve in curves.items():
        frames.append(score_curves(
            sample_id=sample,
            station=station,
            cluster_id=cluster,
            freq=freq,
            log_pred=_logs(curve, clip),
            log_true=log_true,
            log_prior=log_prior,
            method=method,
        ))
    return pd.concat(frames, ignore_index=True)


def _linear_pearson_median(freq, truth, curve) -> float:
    """Loading check for the published linear-|TF| median. Not a harness metric."""

    values = np.array([
        metrics_pearson(np.asarray(curve[i], dtype=np.float64), np.asarray(truth[i], dtype=np.float64))
        for i in range(truth.shape[0])
    ])
    return float(np.median(values))


def _check(
    rows: list[dict],
    *,
    name: str,
    got: float,
    expected: float,
    matched: bool,
    difference: str,
    loading_check: bool = False,
) -> None:
    error = abs(float(got) - float(expected))
    if loading_check:
        status = "loading_check" if error <= AGREE_ABS else "mismatch"
    elif not matched:
        status = "definition_differs"
    elif error <= AGREE_ABS:
        status = "agree"
    else:
        status = "mismatch"
    rows.append({
        "name": name,
        "status": status,
        "harness": float(got),
        "published": float(expected),
        "abs_error": error,
        "difference": difference,
    })


def _base_provenance(sample_ids, **extra) -> dict:
    fields = {
        "label_campaign": extra.pop("label_campaign"),
        "oned_engine": extra.pop("oned_engine"),
        "outcrop_convention": extra.pop("outcrop_convention"),
        "damping_source": extra.pop("damping_source"),
        "fft_window_s": extra.pop("fft_window_s"),
        "checkpoint": extra.pop("checkpoint"),
        "epoch": extra.pop("epoch"),
        "dtype": extra.pop("dtype"),
        "bands": list(BAND_NAMES),
        "aggregation_unit": extra.pop("aggregation_unit"),
        "sample_id_hash": sample_id_hash(sample_ids),
        "git_commit": git_commit(),
        "definition_difference": extra.pop("definition_difference"),
    }
    fields.update(extra)
    return fields


def _epoch_from_history(path: Path) -> int | None:
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    if "epoch" not in frame.columns:
        return None
    if "val_score" in frame.columns:
        scores = pd.to_numeric(frame["val_score"], errors="coerce")
        if scores.notna().any():
            return int(frame.loc[scores.idxmin(), "epoch"])
    return int(frame["epoch"].iloc[-1])


# Station 10 is the center recorder ``center_curves`` and ``fig_practice`` score.
CENTER_STATION = 10
PRACTICE_FAMILIES = {"Horizontal": "iid", "Dipping": "ood_dipping"}


def _center_truth(family: str) -> dict[str, np.ndarray]:
    """Test-split center station from the cache. Anderson's f0 is not loaded."""

    split = np.asarray(np.load(CACHE / family / "split.npy")).astype(str)
    runs = np.flatnonzero(split == "test").astype(np.int64)
    log_tf = np.asarray(np.load(CACHE / family / "log_tf.npy", mmap_mode="r")[runs, CENTER_STATION], dtype=np.float64)
    log_col = np.asarray(np.load(CACHE / family / "log_tf1d.npy", mmap_mode="r")[runs, CENTER_STATION], dtype=np.float64)
    group = np.asarray(np.load(CACHE / family / "group.npy")[runs])
    freq = np.asarray(np.load(CACHE / "freq.npy"), dtype=np.float64)
    return {
        "run": runs,
        "group": group,
        "freq": freq,
        "opensees": np.exp(log_tf),
        "column": np.exp(log_col),
    }


def run_g0(out_dir: Path | None = None) -> dict:
    """Rescore on-disk predictions and write ``G0.md`` plus sidecars."""

    from paths import DUMP

    out_dir = Path(out_dir or ANALYSIS)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = compare_dtype(out_dir=out_dir)
    checks: list[dict] = []
    practice = pd.read_csv(VARDOMAIN / "paper" / "geotech_v2" / "03_state_of_practice" / "table_state_of_practice.csv")
    geotech = VARDOMAIN / "paper" / "geotech_v2"
    long_frames = []
    for domain, family in PRACTICE_FAMILIES.items():
        frame = _center_truth(family)
        matched_path = geotech / "03_state_of_practice" / f"matched_{family}.npz"
        matched = np.load(matched_path)
        order = {int(run): i for i, run in enumerate(np.asarray(matched["run"]))}
        one = np.load(geotech / "07_one_model" / f"center_{family}.npz")
        one_order = {int(run): i for i, run in enumerate(np.asarray(one["run"]))}
        runs = np.asarray(frame["run"])
        curves = {
            "Column": frame["column"],
            "P128n128": _stack(one, "tf", one_order, runs),
            "Pretell": _stack(matched, "pretell", order, runs),
            "Passeri": _stack(matched, "passeri", order, runs),
            "Dmult": _stack(matched, "dmult", order, runs),
            "Toro": _stack(matched, "toro", order, runs),
        }
        scored = _score_family_curves(family, runs, frame["group"], frame["freq"], frame["opensees"], frame["column"], curves, 1e-12)
        scored.insert(0, "domain", domain)
        long_frames.append(scored)
        headline = aggregate(scored, "record")
        for method in curves:
            published = _published_practice(practice, domain, method, "matched Q(Vs) Rayleigh outcrop")
            for band, column in (("mid", "log_l1_0p4_2p5"), ("overall", "log_l1_0p1_10")):
                got = _median_column(headline.frame, method, band, "g_log_l1")
                _check(
                    checks,
                    name=f"{domain} {method} median g log-L1 {band}",
                    got=got,
                    expected=float(published[column]),
                    matched=True,
                    difference="",
                )
            got_pear = _median_column(headline.frame, method, "overall", "pearson_log_tf")
            _check(
                checks,
                name=f"{domain} {method} median Pearson",
                got=got_pear,
                expected=float(published["pearson_record"]),
                matched=False,
                difference=(
                    "README Pearson is metrics.pearson of linear |TF| on the full spectrum, "
                    "then the median across records. The harness Pearson is log|TF| on the named band."
                ),
            )
            loading = _linear_pearson_median(frame["freq"], frame["opensees"], curves[method])
            _check(
                checks,
                name=f"{domain} {method} linear |TF| Pearson loading check",
                got=loading,
                expected=float(published["pearson_record"]),
                matched=True,
                difference="Loading check of the published linear-|TF| median. It is not a harness headline.",
                loading_check=True,
            )
        legacy = _legacy_curves(family, runs)
        legacy_scored = _score_family_curves(
            family, runs, frame["group"], frame["freq"], frame["opensees"], frame["column"], legacy, 1e-12,
        )
        legacy_scored.insert(0, "domain", domain)
        long_frames.append(legacy_scored)
        legacy_head = aggregate(legacy_scored, "record")
        for method in legacy:
            published = _published_practice(practice, domain, method, "cached rigid base")
            for band, column in (("mid", "log_l1_0p4_2p5"), ("overall", "log_l1_0p1_10")):
                got = _median_column(legacy_head.frame, method, band, "g_log_l1")
                _check(
                    checks,
                    name=f"{domain} {method} legacy median g log-L1 {band}",
                    got=got,
                    expected=float(published[column]),
                    matched=True,
                    difference=(
                        "Absolute g log-L1 versus OpenSees matches the stored log-L1 because the prior cancels. "
                        "These curves are the cached rigid-base arms. Skill and harm in the harness use the stored elastic log_tf1d as g=0."
                    ),
                )
        harm = {
            method: {band: harm_rate(scored, band, method) for band in ("mid", "overall")}
            for method in ("P128n128",)
        }
        write_provenance(out_dir / f"practice_{family}.provenance.json", _base_provenance(
            [f"{family}:{int(run)}" for run in runs],
            label_campaign=family,
            oned_engine="stored cache log_tf1d; cache/meta.json fixed is empty, so the engine version is not in that file",
            outcrop_convention="stored log_tf1d for the column; matched arms are an elastic half-space outcrop with Q(Vs) Rayleigh at the record (f1, f2)",
            damping_source="qvs",
            fft_window_s=None,
            checkpoint=str(REPO / "checkpoints" / "vardomain" / "v2_ablations" / "P128n128"),
            epoch=46,
            dtype="float32",
            aggregation_unit="record",
            bootstrap_unit="master",
            definition_difference=(
                "Center station only (station index 10), one row per test record. "
                "README log-L1 is the median of those record scores and uses the same absolute log error. "
                "README Pearson is linear |TF| on the full spectrum. "
                "README harm is the share of records whose linear Pearson or Anderson misfit is worse than the column. "
                "Harness harm is the share of records whose pooled g log-L1 exceeds g=0. "
                f"P128n128 harm_rate mid={harm['P128n128']['mid']:.6g}, overall={harm['P128n128']['overall']:.6g}."
            ),
            n_records=int(runs.size),
            cluster_is="cache group id (Sobol / site)",
        ))
        headline.frame.to_csv(out_dir / f"practice_{family}_record.csv", index=False)
        legacy_head.frame.to_csv(out_dir / f"practice_{family}_legacy_record.csv", index=False)
        checks.append({
            "name": f"{domain} legacy rigid-base prior",
            "status": "definition_differs",
            "harness": float("nan"),
            "published": float("nan"),
            "abs_error": float("nan"),
            "difference": (
                "The legacy Pretell, Passeri, Dmult, and Toro curves are the cached rigid-base arms. "
                "Absolute g log-L1 versus OpenSees still matches the stored log-L1 because the prior cancels. "
                "Harness skill and harm use the stored elastic log_tf1d as g=0, which is a different column from the rigid base those curves were built on."
            ),
        })
        write_provenance(out_dir / f"practice_{family}_legacy.provenance.json", _base_provenance(
            [f"{family}:{int(run)}" for run in runs],
            label_campaign=family,
            oned_engine="stored cache log_tf1d; cache/meta.json fixed is empty, so the engine version is not in that file",
            outcrop_convention="stored elastic log_tf1d is g=0; the curves themselves are rigid-base",
            damping_source="explicit",
            fft_window_s=None,
            checkpoint=None,
            epoch=None,
            dtype="stored |TF|",
            aggregation_unit="record",
            bootstrap_unit="master",
            definition_difference=(
                "Cached rigid-base arms. Toro and Passeri in this set are frequency-independent, at 5% or at the record soil zeta. "
                "Absolute log-L1 versus OpenSees matches the published table. Skill is not taken from that table."
            ),
            n_records=int(runs.size),
        ))

    long_table = pd.concat(long_frames, ignore_index=True)
    long_table.to_csv(out_dir / "practice_long.csv", index=False)

    _rescore_centers(out_dir, checks, "stage2", DUMP / "stage2_center_tf_{family}.npz", {
        "checkpoint": str(REPO / "checkpoints" / "vardomain" / "early_width" / "stage2_rise50"),
        "epoch": 53,
        "dtype": "bfloat16",
        "published_source": "checkpoints/vardomain/early_width/stage2_rise50/eval_test.json",
        "definition_difference": (
            "On-disk spectra are center stations from evaluate.load_model with training autocast (the stage2 sidecar records bfloat16 on CUDA). "
            "The README mid-band skills 0.556, 0.483, and -5.689 are the mean of per-station skills in eval_test.json, "
            "including every station of the 500 m window. That reduction is mean_of_unit_skills. "
            "The harness center table is skill of the pooled center-station error and is a different aggregate. "
            "Three-layer has no center spectra on disk. Stage 1 has no spectra on disk."
        ),
    })
    _rescore_centers(out_dir, checks, "ensemble", DUMP / "ensemble_center_tf_{family}.npz", {
        "checkpoint": "mean of g from P128 epoch 58, P128s1 epoch 74, P128n128 epoch 46, P128c epoch 30",
        "epoch": None,
        "dtype": "float32",
        "published_source": "paper/dump/ensemble_eval_test.json",
        "definition_difference": (
            "The stored center |TF| is the float32 mean of g from ensemble_update._mean_g, saved as exp(log prior + mean g). "
            "The README skills 0.596 (horizontal 500 m) and 0.471 (three-layer 500 m) are skill of pooled log-L1 over every station. "
            "This table is the center station only. Three-layer center spectra for the full test split are not on disk. "
            "Live _mean_g was not re-run."
        ),
    })
    _treasure(out_dir, checks)
    _unscored_bins(out_dir, checks)
    _stage_mean_versus_pooled(out_dir, checks)
    report = {
        "dtype": dtype,
        "checks": checks,
        "g0_holds": all(row["status"] != "mismatch" for row in checks),
    }
    (out_dir / "g0_checks.json").write_text(json.dumps(report, indent=2) + "\n")
    (out_dir / "G0.md").write_text(_g0_markdown(report))
    return report


def _legacy_curves(family: str, runs: np.ndarray) -> dict[str, np.ndarray]:
    from paths import FIG

    arms = np.load(FIG / f"test_arms_{family}.npz")
    order = {int(run): i for i, run in enumerate(np.asarray(arms["run"]))}
    passeri = np.load(FIG / f"test_passeri_{family}.npz")
    porder = {int(run): i for i, run in enumerate(np.asarray(passeri["run"]))}
    zeta = np.load(FIG / f"test_zeta_arms_{family}.npz")
    zorder = {int(run): i for i, run in enumerate(np.asarray(zeta["run"]))}

    def take(blob, key, index):
        return np.stack([np.asarray(blob[key][index[int(run)]], dtype=np.float64) for run in runs])

    return {
        "Pretell": take(arms, "pretell", order),
        "Toro": take(arms, "toro", order),
        "Dmult": take(arms, "dmult", order),
        "Passeri": take(passeri, "passeri", porder),
        "Toro (record zeta, rigid base)": take(zeta, "toro", zorder),
        "Passeri (record zeta, rigid base)": take(zeta, "passeri", zorder),
    }


def _rescore_centers(out_dir: Path, checks: list[dict], tag: str, pattern: Path, meta: dict) -> None:
    frames = []
    ids: list[str] = []
    for domain, family in PRACTICE_FAMILIES.items():
        path = Path(str(pattern).format(family=family))
        if not path.is_file():
            checks.append({
                "name": f"{tag} {domain} center spectra",
                "status": "definition_differs",
                "harness": float("nan"),
                "published": float("nan"),
                "abs_error": float("nan"),
                "difference": f"{path} is absent, so this family was not rescored from curves.",
            })
            continue
        blob = np.load(path)
        frame = _center_truth(family)
        order = {int(run): i for i, run in enumerate(np.asarray(blob["run"]))}
        runs = np.asarray([int(run) for run in frame["run"] if int(run) in order], dtype=np.int64)
        missing = int(frame["run"].size - runs.size)
        curves = {tag: _stack(blob, "tf", order, runs)}
        group = []
        truth_rows = []
        column_rows = []
        run_set = {int(run): i for i, run in enumerate(frame["run"])}
        for run in runs:
            src = run_set[int(run)]
            group.append(frame["group"][src])
            truth_rows.append(frame["opensees"][src])
            column_rows.append(frame["column"][src])
        scored = _score_family_curves(
            family, runs, np.asarray(group), frame["freq"],
            np.stack(truth_rows), np.stack(column_rows), curves, 1e-12,
        )
        scored.insert(0, "domain", domain)
        frames.append(scored)
        ids.extend(f"{family}:{int(run)}" for run in runs)
        headline = aggregate(scored, "record")
        headline.frame.to_csv(out_dir / f"{tag}_{family}_record.csv", index=False)
        published = _published_bin(tag, family)
        got = _median_column(headline.frame, tag, "mid", "g_log_l1")
        _check(
            checks,
            name=f"{tag} {domain} center median g log-L1 mid",
            got=got,
            expected=float(published["log_l1_mid"]),
            matched=False,
            difference=(
                f"{meta['definition_difference']} Missing test runs relative to the cache split: {missing}. "
                f"Published all-station log_l1_mid={published['log_l1_mid']}, skill_mid={published['skill_mid']}, "
                f"n_stations={published['n_stations']}. Harness center median g log-L1 mid={got}."
            ),
        )
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(out_dir / f"{tag}_long.csv", index=False)
    write_provenance(out_dir / f"{tag}_center.provenance.json", _base_provenance(
        ids or ["none"],
        label_campaign="iid and ood_dipping test split, center station",
        oned_engine="stored cache log_tf1d; cache/meta.json fixed is empty, so the engine version is not in that file",
        outcrop_convention="stored log_tf1d",
        damping_source="qvs",
        fft_window_s=None,
        checkpoint=meta["checkpoint"],
        epoch=meta["epoch"],
        dtype=meta["dtype"],
        aggregation_unit="record",
        bootstrap_unit="master",
        definition_difference=meta["definition_difference"],
        published_source=meta["published_source"],
    ))


def _published_bin(tag: str, family: str) -> dict:
    if tag == "stage2":
        path = REPO / "checkpoints" / "vardomain" / "early_width" / "stage2_rise50" / "eval_test.json"
        block = json.loads(path.read_text())["bins"][f"{family}_w500"]
        return {"log_l1_mid": float(block["log_l1_mid"]), "skill_mid": float(block["skill_mid"]), "n_stations": int(block["n_stations"])}
    path = VARDOMAIN / "paper" / "dump" / "ensemble_eval_test.json"
    block = json.loads(path.read_text())["bins"][f"{family}_w500"]
    return {"log_l1_mid": float(block["log_l1_mid"]), "skill_mid": float(block["skill_mid"]), "n_stations": int(block["n_stations"])}


def _treasure(out_dir: Path, checks: list[dict]) -> None:
    blob = np.load(VARDOMAIN / "paper" / "geotech_v2" / "05_treasure_island_full" / "fields.npz")
    published = json.loads(
        (VARDOMAIN / "paper" / "geotech_v2" / "05_treasure_island_full" / "treasure_island_full_metrics.json").read_text()
    )
    freq = np.asarray(blob["freq"], dtype=np.float64)
    truth = np.asarray(blob["truth"], dtype=np.float64)
    column = np.asarray(blob["column"], dtype=np.float64)
    x = np.asarray(blob["x"], dtype=np.float64)
    one_g = np.load(VARDOMAIN / "paper" / "geotech_v2" / "07_one_model" / "treasure_g.npz")
    if not np.allclose(np.asarray(one_g["x"]), x):
        raise ValueError("treasure_g.npz stations do not match fields.npz")
    # fields.npz pred is the four-checkpoint curve. The README one-model number
    # is column * exp(g) from 07_one_model/treasure_g.npz.
    pred = column * np.exp(np.asarray(one_g["g"], dtype=np.float64))
    n = pred.shape[0]
    # The stored figure clipped |TF| at 1e-6 before the log. Score that input,
    # and also the stored g, and keep the larger station-wise gap.
    clipped = score_curves(
        sample_id=np.array(["treasure_island"] * n),
        station=np.arange(n),
        cluster_id=np.array(["treasure_island"] * n),
        freq=freq,
        log_pred=_logs(pred, 1e-6),
        log_true=_logs(truth, 1e-6),
        log_prior=_logs(column, 1e-6),
        method="P128n128",
        x_m=x,
    )
    clipped.to_csv(out_dir / "treasure_long.csv", index=False)
    station = aggregate(clipped, "station")
    for band, key in (("mid", "median_log_l1_0p4_2p5"), ("overall", "median_log_l1_0p1_10")):
        got = _median_column(station.frame, "P128n128", band, "g_log_l1")
        expected = float(published[key]) if key in published else float("nan")
        if key not in published and band == "overall":
            expected = float(np.median(np.asarray(
                pd.read_csv(VARDOMAIN / "paper" / "geotech_v2" / "05_treasure_island_full" / "treasure_island_stations.csv")["log_l1_0p1_10"]
            )))
        _check(
            checks,
            name=f"Treasure Island median station g log-L1 {band}",
            got=got,
            expected=expected,
            matched=True,
            difference="",
        )
    got_pear = _median_column(station.frame, "P128n128", "mid", "pearson_log_tf")
    _check(
        checks,
        name="Treasure Island median Pearson of log|TF| mid",
        got=got_pear,
        expected=float(published["median_pearson_log_0p4_2p5"]),
        matched=True,
        difference="",
    )
    pooled = aggregate(clipped, "record")
    pooled_skill = float(pooled.frame.loc[pooled.frame["band"] == "mid", "skill"].iloc[0])
    _check(
        checks,
        name="Treasure Island skill",
        got=pooled_skill,
        expected=float(published["median_skill_0p4_2p5"]),
        matched=False,
        difference=(
            "README skill is the median of per-station skills with a 1e-6 denominator. "
            "The harness headline on one section is the skill of the pooled station error, "
            "suppressed where L1(g_true) is at or below the noise-floor placeholder. "
            "Those two reductions are not compared."
        ),
    )
    direct = np.array([
        g_log_l1(np.asarray(one_g["g"])[i], np.asarray(blob["g_true"])[i], freq, "mid")
        for i in range(n)
    ])
    clipped_mid = clipped.loc[(clipped["band"] == "mid") & (clipped["metric"] == "g_log_l1"), "value"].to_numpy()
    clip_gap = float(np.max(np.abs(direct - clipped_mid)))
    mid = station.frame.loc[station.frame["band"] == "mid"].sort_values("station")
    block = bootstrap_moving_block(
        mid["g_log_l1"].to_numpy(),
        x[mid["station"].to_numpy().astype(int)],
        rng=np.random.default_rng(0),
    )
    write_provenance(out_dir / "treasure_p128n128.provenance.json", _base_provenance(
        ["treasure_island"],
        label_campaign="case7a full_qvs, one 4000 m section",
        oned_engine="section_1d.surface_over_outcrop via matched_physics.outcrop_many; half-space outcrop",
        outcrop_convention="surface over elastic outcrop",
        damping_source="qvs",
        fft_window_s=1310.72,
        checkpoint=str(REPO / "checkpoints" / "vardomain" / "v2_ablations" / "P128n128"),
        epoch=46,
        dtype="float32",
        aggregation_unit="station",
        bootstrap_unit="moving block along x",
        definition_difference=(
            "The 15 s OpenSees record is one section. The quoted log-L1 and the Pearson of log|TF| match the harness station metrics. "
            "fields.npz pred is the four-checkpoint curve; the one-model g is 07_one_model/treasure_g.npz. "
            f"Clipping |TF| at 1e-6 before the log moves mid-band station g log-L1 by at most {clip_gap:.3e} relative to the stored g. "
            "SVLAB appendix Pearson is linear |TF| on 0.1–2.5 Hz and is not a harness headline. "
            f"Moving-block median of mid-band station g log-L1: point {block['point']:.6g}, interval {block['low']:.6g}–{block['high']:.6g}, block_m {block['block_m']}."
        ),
        record_s=15.0,
        n_stations=n,
    ))


def _unscored_bins(out_dir: Path, checks: list[dict]) -> None:
    one = pd.read_csv(VARDOMAIN / "paper" / "geotech_v2" / "07_one_model" / "skill_by_bin.csv")
    row = one.loc[one["bin"] == "iid_w500"].iloc[0]
    checks.append({
        "name": "P128n128 horizontal 500 m all-station pooled skill",
        "status": "definition_matches_not_recomputed",
        "harness": float("nan"),
        "published": float(row["skill_mid"]),
        "abs_error": float("nan"),
        "difference": (
            "skill_by_bin.csv skill_mid is 1 - (mean station log-L1) / (mean station column log-L1) "
            f"from diag_forward.load_fp32, epoch 46, float32. Published skill_mid={float(row['skill_mid'])}, "
            f"log_l1_mid={float(row['log_l1_mid'])}, n_stations={int(row['n_stations'])}. "
            "That is the harness pooled reduction. The 15120 station spectra are not on disk, so this row was not recomputed. "
            "The center-station log-L1 in the practice table is the independent check for this checkpoint. "
            "Stage 1 eval_test.json is the mean of per-station skills under training autocast, and no stage-1 spectra are on disk. "
            "The README sentence 0.616 → 0.556 → 0.548 chains that mean-of-skills with this pooled skill."
        ),
    })
    stage1 = json.loads((REPO / "checkpoints" / "vardomain" / "early_width" / "stage1" / "eval_test.json").read_text())
    block = stage1["bins"]["iid_w500"]
    three = json.loads((REPO / "checkpoints" / "vardomain" / "early_width" / "stage2_rise50" / "eval_test.json").read_text())
    three_block = three["bins"]["ood_three_layer_w500"]
    checks.append({
        "name": "stage 1 horizontal 500 m mean of station skills",
        "status": "definition_differs",
        "harness": float("nan"),
        "published": float(block["skill_mid"]),
        "abs_error": float("nan"),
        "difference": (
            "Stage 1 has no saved spectra. eval_test.json skill_mid is the mean of per-station skills "
            f"({float(block['skill_mid'])}), log_l1_mid={float(block['log_l1_mid'])}, n_stations={int(block['n_stations'])}. "
            "config amp is true. The harness does not treat this as pooled g log-L1."
        ),
    })
    checks.append({
        "name": "stage 2 three-layer mean of station skills",
        "status": "definition_differs",
        "harness": float("nan"),
        "published": float(three_block["skill_mid"]),
        "abs_error": float("nan"),
        "difference": (
            "The README value -5.689 is eval_test.json skill_mid for ood_three_layer_w500, "
            f"the mean of per-station skills ({float(three_block['skill_mid'])}), "
            f"with log_l1_mid={float(three_block['log_l1_mid'])}. "
            "No three-layer center spectra are on disk. "
            "A pooled skill is suppressed when L1(g_true) is at or below the noise-floor placeholder; "
            "this bin's log-L1 sits near that placeholder and the published skill was not suppressed."
        ),
    })
    write_provenance(out_dir / "p128n128_all_station_bin.provenance.json", _base_provenance(
        ["iid_w500"],
        label_campaign="iid test, all stations, 500 m",
        oned_engine="stored cache log_tf1d; cache/meta.json fixed is empty, so the engine version is not in that file",
        outcrop_convention="stored log_tf1d",
        damping_source="qvs",
        fft_window_s=None,
        checkpoint=str(REPO / "checkpoints" / "vardomain" / "v2_ablations" / "P128n128"),
        epoch=46,
        dtype="float32",
        aggregation_unit="station",
        bootstrap_unit="not recomputed",
        definition_difference=(
            "Pooled skill of all stations matches the harness reduction and the float32 loader. "
            "Station spectra are absent, so the stored bin was not recomputed. "
            "aggregation_unit here names the stations that entered the pool; the skill itself is skill_of_pooled, not the mean of station skills."
        ),
        skill_reduction=SKILL_POOLED,
        published_skill_mid=float(row["skill_mid"]),
        published_log_l1_mid=float(row["log_l1_mid"]),
    ))
    write_provenance(out_dir / "stage1_eval_test.provenance.json", _base_provenance(
        ["iid_w500", "ood_dipping_w500", "ood_three_layer_w500"],
        label_campaign="iid, ood_dipping, ood_three_layer test",
        oned_engine="stored cache log_tf1d; cache/meta.json fixed is empty, so the engine version is not in that file",
        outcrop_convention="stored log_tf1d",
        damping_source="qvs",
        fft_window_s=None,
        checkpoint=str(REPO / "checkpoints" / "vardomain" / "early_width" / "stage1"),
        epoch=_epoch_from_history(REPO / "checkpoints" / "vardomain" / "early_width" / "stage1" / "history.csv"),
        dtype="bfloat16",
        aggregation_unit="station",
        bootstrap_unit="not recomputed",
        definition_difference=(
            "Mean of per-station skills from evaluate.py under training autocast. "
            "No spectra are on disk. The harness does not recompute this file and does not compare it with a pooled skill."
        ),
        skill_reduction=SKILL_MEAN,
        published_iid_w500_skill_mid=float(block["skill_mid"]),
    ))


def _stage_mean_versus_pooled(out_dir: Path, checks: list[dict]) -> None:
    """Show the stored mean-of-skills and the stored pooled skill side by side, without comparing them."""

    records = pd.read_csv(
        VARDOMAIN / "paper" / "dump" / "records.csv",
        usecols=["pass", "family", "width", "mean_skill_mid"],
    )
    part = records[(records["pass"] == "ensemble") & (records["family"] == "iid") & (records["width"] == 500)]
    mean_of_means = float(part["mean_skill_mid"].mean()) if len(part) else float("nan")
    pooled = float(json.loads((VARDOMAIN / "paper" / "dump" / "ensemble_eval_test.json").read_text())["bins"]["iid_w500"]["skill_mid"])
    try:
        assert_comparable(
            Headline(pd.DataFrame(), "station", SKILL_MEAN),
            Headline(pd.DataFrame(), "station", SKILL_POOLED),
        )
        refused = False
    except AggregationError:
        refused = True
    checks.append({
        "name": "ensemble horizontal 500 m mean-of-skills beside pooled skill",
        "status": "definition_differs",
        "harness": mean_of_means,
        "published": pooled,
        "abs_error": abs(mean_of_means - pooled),
        "difference": (
            f"records.csv mean of per-record mean_skill_mid is {mean_of_means} (n={int(len(part))}). "
            f"ensemble_eval_test.json pooled skill_mid is {pooled}. "
            f"assert_comparable refuses to put mean_of_unit_skills beside pooled: {refused}. "
            "The two numbers are both stored and are not a harness agreement check."
        ),
    })
    (out_dir / "ensemble_reductions.json").write_text(json.dumps({
        "mean_of_record_mean_skill_mid": mean_of_means,
        "pooled_skill_mid": pooled,
        "n_records": int(len(part)),
        "comparison_refused": refused,
    }, indent=2) + "\n")


def _g0_markdown(report: dict) -> str:
    dtype = report["dtype"]
    lines = [
        "# G0",
        "",
        "Phase 0 only. No later phase is started here.",
        "",
        "Agreement within 1e-6 is against the unrounded CSV and JSON values that `geotech_v2/README.md` rounds. The README text itself is rounded to three to five digits, so a 1e-6 check against the printed rounding would fail on a value the scorer reproduced.",
        "",
        f"G0 holds: {report['g0_holds']}. A hold means every recomputed quantity whose definition matches is within 1e-6, and every other README quantity has the difference named. `definition_matches_not_recomputed` means the stored bin already used pooled float32 log-L1 and the station spectra are not on disk.",
        "",
        "## Dtype",
        "",
        f"Window: {dtype['scope']}. IID local index {dtype['local_index']}, group {dtype['group_id']}, {dtype['n_stations']} stations, device `{dtype['device']}`.",
        "",
        f"Maximum absolute g log-L1 between `load_fp32` and `evaluate.load_model`: {dtype['max_abs_g_log_l1']:.3e}. Noise-floor placeholder: {dtype['noise_floor_placeholder']}. Frozen dtype: {dtype['frozen_dtype']}.",
        "",
        dtype.get("autocast_note", "Training autocast ran."),
        "",
        "Per band: " + ", ".join(f"{band} {value:.3e}" for band, value in dtype["max_abs_g_log_l1_by_band"].items()) + ".",
        "",
        "## Checks",
        "",
        "| Status | Name | Harness | Published | Abs error |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report["checks"]:
        harness = "" if row["harness"] != row["harness"] else f"{row['harness']:.8g}"
        published = "" if row["published"] != row["published"] else f"{row['published']:.8g}"
        error = "" if row["abs_error"] != row["abs_error"] else f"{row['abs_error']:.3e}"
        lines.append(f"| {row['status']} | {row['name']} | {harness} | {published} | {error} |")
    lines.extend(["", "## Where the definition differs", ""])
    for row in report["checks"]:
        if row["status"] in ("agree", "loading_check"):
            continue
        lines.append(f"- **{row['name']}.** {row['difference']}")
    lines.extend([
        "",
        "## What a later phase calls",
        "",
        "- `band_mask`, `g_log_l1`, `prior_log_l1`, `skill_value`, `pearson_log_tf` score one curve. `g_log_l1` and `pearson_log_tf` call `metrics.log_l1` and `metrics.pearson`.",
        "- `score_curves` writes the long table. Columns: `sample_id`, `station`, `cluster_id`, `x_m`, `method`, `band`, `metric`, `value`. Metrics: `g_log_l1`, `prior_log_l1`, `skill`, `pearson_log_tf`, `harmed`. Bands: `low`, `mid`, `high`, `overall`.",
        "- `aggregate(table, unit)` with `unit` in `record`, `station`, `master` returns a `Headline` whose skill is `pooled`. `aggregate_mean_skills` is the other reduction. `assert_comparable` raises `AggregationError` if the units or the reductions differ.",
        "- `harm_rate(table, band, method)` is the share of `sample_id` values whose pooled g log-L1 exceeds g = 0.",
        "- `bootstrap_independent(..., unit='record'|'master')` resamples that unit through `bootstrap_ci.bootstrap_statistic`. `unit='station'` and `unit='frequency'` raise. One section uses `bootstrap_moving_block`, and the sidecar labels that unit `moving block along x`.",
        "- `write_provenance` requires the Phase 0 fields: label campaign, 1D engine, outcrop convention, damping source, FFT window, checkpoint, epoch, dtype, bands, aggregation unit, sample-id hash, git commit.",
        "- `compare_dtype` is the dtype entry point. `run_g0` writes this note.",
        "- The active noise floor is `gates.noise_floor_log_l1()`. It is the placeholder 0.0136 until Phase 1.4 measures `max(FE floor, 1D-convention uncertainty)`. `metrics.FLOOR_LOG_L1` is not that floor.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    report = run_g0(ANALYSIS)
    print(f"g0_holds {report['g0_holds']} checks {len(report['checks'])} frozen {report['dtype']['frozen_dtype']}", flush=True)


if __name__ == "__main__":
    main()
