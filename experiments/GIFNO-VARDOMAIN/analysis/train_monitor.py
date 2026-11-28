#!/usr/bin/env python3
"""Epoch log for a future training run. This module does not start one.

It scores a harness long table and reads the Phase 1 audit floors. Each call
appends overall and per-band ``g`` log-L1 by family, the harm rate, the
consistency probes, and the gap to the POD and realization floors.

Blind rows are written to ``blind_sealed.csv``. ``read_sealed_blind`` raises
before opening that file unless ``open_blind`` is true.

Stop, from ``gates.py``: one consistency probe is false for
``CONSISTENCY_FAIL_EPOCHS`` epochs, or the overall-band harm rate exceeds
``HARM_RATE_STOP`` in any family.

Accept versus P128n128: overall validation log-L1 improves by more than the
bootstrap half-width, no family is worse by more than ``TOST_DELTA_LOG_L1``,
the homogeneous, CoV = 0, and mirror probes pass, and at least
``MIN_TRAINING_SEEDS`` seeds have been trained. This file does not train them.

    python train_monitor.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_ANALYSIS = Path(__file__).resolve().parent
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

import gates  # noqa: E402
import score_harness  # noqa: E402

AUDIT = _ANALYSIS / "audit"
SEALED_NAME = "blind_sealed.csv"
MONITOR_NAME = "monitor.csv"
SEAL_NOTE = "sealed blind; read_sealed_blind requires open_blind=True"
ACCEPT_PROBES = ("homogeneous_column", "cov0", "mirror")
MONITOR_COLUMNS = (
    "epoch",
    "split",
    "family",
    "band",
    "g_log_l1",
    "harm_rate",
    "n_records",
    "pod_floor",
    "realization_floor",
    "gap_pod",
    "gap_realization",
    "bootstrap_half_width",
    "homogeneous_column",
    "cov0",
    "mirror",
    "consistency_failed",
    "consistency_fail_streak",
    "stop",
    "stop_reason",
    "n_seeds",
    "improvement",
    "accept_half_width",
    "family_margin",
    "accept",
    "accept_blocked",
)


class BlindSealed(RuntimeError):
    """The blind file was not opened."""


@dataclass(frozen=True)
class Floors:
    """POD reconstruction floor and the median realization spread, by family and band."""

    pod: dict[tuple[str, str], float]
    realization: dict[tuple[str, str], float]

    def pod_at(self, family: str, band: str) -> float:
        return float(self.pod.get((family, band), np.nan))

    def realization_at(self, family: str, band: str) -> float:
        return float(self.realization.get((family, band), np.nan))


def load_audit_floors(audit_dir: Path | None = None, split: str = "val") -> Floors:
    """Read ``representation.csv`` and ``realization.csv``. Does not write them.

    The POD entry is the ``family:split`` pool. The realization entry is the
    median over Sobol groups of the pairwise ``|g|`` spread. Manifold sections
    have no realization rows.
    """

    root = Path(audit_dir or AUDIT)
    rep = pd.read_csv(root / "representation.csv")
    pod: dict[tuple[str, str], float] = {}
    suffix = ":" + str(split)
    floors = rep.loc[rep["table"].astype(str) == "pod_floor"]
    for row in floors.itertuples(index=False):
        pool = str(row.pool)
        if not pool.endswith(suffix):
            continue
        family = pool[: -len(suffix)]
        if not family or ":" in family:
            continue
        pod[(family, str(row.band))] = float(row.pod_floor_l1)
    real = pd.read_csv(root / "realization.csv")
    real = real.loc[real["table"].astype(str) == "realization"]
    realization: dict[tuple[str, str], float] = {}
    if not real.empty:
        grouped = real.groupby(["family", "band"], sort=False)["value"].median()
        for (family, band), value in grouped.items():
            realization[(str(family), str(band))] = float(value)
    return Floors(pod, realization)


def write_sealed_blind(frame: pd.DataFrame, path: Path) -> None:
    """Write blind rows. The open monitor CSV does not receive them."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = frame.to_csv(index=False)
    path.write_text(f"# {SEAL_NOTE}\n{body}")


def read_sealed_blind(path: Path, *, open_blind: bool = False) -> pd.DataFrame:
    """Open the sealed file only when ``open_blind`` is true.

    The flag is checked before the path is touched. The default refuses a
    missing file with ``BlindSealed`` rather than ``FileNotFoundError``.
    """

    if not open_blind:
        raise BlindSealed(
            "Blind scores stay sealed until acceptance. Pass open_blind=True to read them."
        )
    return pd.read_csv(Path(path), comment="#")


def consistency_fail_streak(history: list[dict[str, bool]]) -> int:
    """Longest trailing run of one named probe being false.

    A probe missing from an epoch breaks that probe's streak. Different probes
    failing on different epochs do not add together.
    """

    names: list[str] = []
    for item in history:
        for name in item:
            if name not in names:
                names.append(name)
    best = 0
    for name in names:
        streak = 0
        for item in reversed(history):
            if item.get(name) is not False:
                break
            streak += 1
        best = max(best, streak)
    return best


def stop_decision(history: list[dict[str, bool]], harm_by_family: dict[str, float]) -> dict:
    """Proposed stop. Harm is the overall-band rate; per-band harm is only logged.

    Exceeding ``HARM_RATE_STOP`` is strict. A rate equal to 5% does not stop.
    """

    streak = consistency_fail_streak(history)
    harm_hit = {
        str(family): float(rate)
        for family, rate in harm_by_family.items()
        if np.isfinite(rate) and float(rate) > gates.HARM_RATE_STOP
    }
    reasons = []
    if streak >= gates.CONSISTENCY_FAIL_EPOCHS:
        reasons.append(f"a consistency test failed for {streak} epochs")
    if harm_hit:
        names = ", ".join(f"{family}={harm_hit[family]:.3f}" for family in sorted(harm_hit))
        reasons.append(f"harm rate exceeds {gates.HARM_RATE_STOP:.0%} in {names}")
    return {
        "stop": bool(reasons),
        "consistency_fail_streak": streak,
        "harm_families": harm_hit,
        "reason": "; ".join(reasons),
    }


def consistency_passed_for_accept(probes: dict[str, bool]) -> bool:
    """Homogeneous column, CoV = 0, and mirror are the accept probes.

    Monotone pairs are not required. No finite-element pair for that probe
    exists, and the frozen table left it unrun.
    """

    return all(probes.get(name) is True for name in ACCEPT_PROBES)


def accept_versus_p128(
    *,
    overall_val: float,
    baseline_overall: float,
    half_width: float,
    family_val: dict[str, float],
    family_baseline: dict[str, float],
    consistency_passed: bool,
    n_seeds: int,
    margin: float | None = None,
) -> dict:
    """Proposed accept rule against P128n128. Lower log-L1 is the improvement.

    A family delta equal to the margin is not worse by more than the margin.
    The improvement must be strictly larger than the bootstrap half-width.
    """

    margin = gates.TOST_DELTA_LOG_L1 if margin is None else float(margin)
    improvement = float(baseline_overall) - float(overall_val)
    improved = bool(np.isfinite(improvement) and np.isfinite(half_width) and improvement > float(half_width))
    deltas = {}
    missing = []
    for family, value in family_val.items():
        if family not in family_baseline:
            missing.append(family)
            continue
        deltas[family] = float(value) - float(family_baseline[family])
    family_ok = bool(family_val) and not missing and all(delta <= margin + 1e-12 for delta in deltas.values())
    seeds_ok = int(n_seeds) >= gates.MIN_TRAINING_SEEDS
    consistency_ok = bool(consistency_passed)
    reasons = []
    if not improved:
        reasons.append("overall improvement is not larger than the bootstrap half-width")
    if not family_ok:
        reasons.append("a family is worse by more than the margin, or has no baseline")
    if not consistency_ok:
        reasons.append("a consistency test did not pass")
    if not seeds_ok:
        reasons.append(f"fewer than {gates.MIN_TRAINING_SEEDS} training seeds")
    return {
        "accept": bool(improved and family_ok and consistency_ok and seeds_ok),
        "improvement": improvement,
        "half_width": float(half_width),
        "margin": margin,
        "family_delta": deltas,
        "n_seeds": int(n_seeds),
        "blocked_by": reasons,
    }


def _gap(model: float, floor: float) -> float:
    if not np.isfinite(model) or not np.isfinite(floor):
        return float("nan")
    return float(model) - float(floor)


def _probe_cell(probes: dict[str, bool], name: str) -> str:
    if name not in probes:
        return ""
    return "pass" if probes[name] else "fail"


def _half_width(values: np.ndarray, unit_ids: np.ndarray, *, n_draw: int, rng: np.random.Generator) -> float:
    if values.size == 0:
        return float("nan")
    out = score_harness.bootstrap_independent(
        values, unit_ids, unit="record", statistic=np.mean, n_draw=n_draw, rng=rng,
    )
    return 0.5 * (float(out["high"]) - float(out["low"]))


def _record_means(part: pd.DataFrame) -> pd.Series:
    rows = part.loc[part["metric"] == "g_log_l1"]
    if rows.empty:
        return pd.Series(dtype=np.float64)
    return rows.groupby("sample_id", sort=False)["value"].mean()


def _require_harness_table(table: pd.DataFrame) -> None:
    needed = set(score_harness.LONG_COLUMNS) | {"family", "split"}
    missing = sorted(needed - set(table.columns))
    if missing:
        raise ValueError(f"monitor table is missing {missing}; pass a harness long table with family and split")


def log_epoch(
    table: pd.DataFrame,
    *,
    epoch: int,
    out_dir: Path,
    floors: Floors,
    consistency: dict[str, bool],
    consistency_history: list[dict[str, bool]] | None = None,
    baseline_overall: float | None = None,
    baseline_by_family: dict[str, float] | None = None,
    n_seeds: int = 0,
    method: str = "candidate",
    blind_split: str = "blind",
    n_draw: int = 200,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Append one epoch to ``monitor.csv`` and rewrite the sealed blind file.

    ``consistency_history`` is the previous epochs only. This call appends the
    current probes when it measures the stop streak. Blind ``sample_id`` values
    are absent from the frame that is returned and written to the open CSV.
    """

    _require_harness_table(table)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng() if rng is None else rng
    blind = table.loc[table["split"].astype(str) == str(blind_split)].copy()
    visible = table.loc[table["split"].astype(str) != str(blind_split)].copy()
    write_sealed_blind(blind, out_dir / SEALED_NAME)
    scored = visible.loc[visible["method"].astype(str) == str(method)].copy()
    if scored.empty:
        raise ValueError(f"no rows for method {method!r}")
    val = scored.loc[scored["split"].astype(str) == "val"]
    ruled = val if not val.empty else scored
    history = list(consistency_history or [])
    history.append(dict(consistency))
    families = list(dict.fromkeys(ruled["family"].astype(str)))
    harm_overall = {}
    for family in families:
        part = ruled.loc[ruled["family"].astype(str) == family]
        if part.loc[(part["band"] == "overall") & (part["metric"] == "g_log_l1")].empty:
            continue
        harm_overall[family] = score_harness.harm_rate(part, "overall", method)
    decision = stop_decision(history, harm_overall)
    failed = ",".join(name for name, passed in consistency.items() if passed is False)
    family_val = {}
    for family in families:
        means = _record_means(ruled.loc[(ruled["family"].astype(str) == family) & (ruled["band"] == "overall")])
        if not means.empty:
            family_val[family] = float(means.mean())
    pooled = _record_means(ruled.loc[ruled["band"] == "overall"])
    accept_half = _half_width(pooled.to_numpy(), pooled.index.to_numpy(), n_draw=n_draw, rng=rng)
    if baseline_overall is None or not family_val:
        accepted = accept_versus_p128(
            overall_val=float("nan"),
            baseline_overall=float("nan"),
            half_width=accept_half,
            family_val={},
            family_baseline={},
            consistency_passed=False,
            n_seeds=n_seeds,
        )
        accepted["blocked_by"] = ["no P128n128 baseline was passed"]
        accepted["accept"] = False
        accepted["improvement"] = float("nan")
    else:
        accepted = accept_versus_p128(
            overall_val=float(pooled.mean()) if not pooled.empty else float("nan"),
            baseline_overall=float(baseline_overall),
            half_width=accept_half,
            family_val=family_val,
            family_baseline=dict(baseline_by_family or {}),
            consistency_passed=consistency_passed_for_accept(consistency),
            n_seeds=n_seeds,
        )
    rows = []
    for (split, family, band), part in scored.groupby(["split", "family", "band"], sort=False):
        means = _record_means(part)
        if means.empty:
            continue
        model = float(means.mean())
        rate = score_harness.harm_rate(part, str(band), method)
        pod = floors.pod_at(str(family), str(band))
        real = floors.realization_at(str(family), str(band))
        half = _half_width(means.to_numpy(), means.index.to_numpy(), n_draw=n_draw, rng=rng)
        rows.append({
            "epoch": int(epoch),
            "split": str(split),
            "family": str(family),
            "band": str(band),
            "g_log_l1": model,
            "harm_rate": rate,
            "n_records": int(means.shape[0]),
            "pod_floor": pod,
            "realization_floor": real,
            "gap_pod": _gap(model, pod),
            "gap_realization": _gap(model, real),
            "bootstrap_half_width": half,
            "homogeneous_column": _probe_cell(consistency, "homogeneous_column"),
            "cov0": _probe_cell(consistency, "cov0"),
            "mirror": _probe_cell(consistency, "mirror"),
            "consistency_failed": failed,
            "consistency_fail_streak": int(decision["consistency_fail_streak"]),
            "stop": "yes" if decision["stop"] else "no",
            "stop_reason": decision["reason"],
            "n_seeds": int(n_seeds),
            "improvement": accepted["improvement"],
            "accept_half_width": accept_half,
            "family_margin": accepted["margin"],
            "accept": "yes" if accepted["accept"] else "no",
            "accept_blocked": "; ".join(accepted["blocked_by"]),
        })
    frame = pd.DataFrame(rows, columns=list(MONITOR_COLUMNS))
    path = out_dir / MONITOR_NAME
    frame.to_csv(path, mode="a", header=not path.is_file(), index=False)
    return frame


def synthetic_table(seed: int = 0) -> pd.DataFrame:
    """Four validation records and one blind record, scored by the harness."""

    rng = np.random.default_rng(seed)
    freq = np.geomspace(0.1, 10.0, 16)
    specs = (
        ("iid-0", "g0", "iid", "val"),
        ("iid-1", "g1", "iid", "val"),
        ("man-0", "m0", "manifold", "val"),
        ("man-1", "m1", "manifold", "val"),
        ("blind-0", "b0", "manifold", "blind"),
    )
    n = len(specs)
    log_true = rng.normal(scale=0.05, size=(n, freq.size))
    # Fixed gaps keep every record inside g = 0, so the dry-run harm rate is zero.
    log_prior = log_true + 0.2
    log_pred = log_true + 0.02
    table = score_harness.score_curves(
        sample_id=np.array([item[0] for item in specs]),
        station=np.zeros(n, dtype=int),
        cluster_id=np.array([item[1] for item in specs]),
        freq=freq,
        log_pred=log_pred,
        log_true=log_true,
        log_prior=log_prior,
        method="candidate",
    )
    family = {item[0]: item[2] for item in specs}
    split = {item[0]: item[3] for item in specs}
    table["family"] = table["sample_id"].map(family)
    table["split"] = table["sample_id"].map(split)
    return table


def dry_run(out_dir: Path, audit_dir: Path | None = None) -> pd.DataFrame:
    """One synthetic epoch. Reads the audit floors and does not train."""

    table = synthetic_table()
    floors = load_audit_floors(audit_dir, split="val")
    return log_epoch(
        table,
        epoch=0,
        out_dir=out_dir,
        floors=floors,
        consistency={"homogeneous_column": True, "cov0": True, "mirror": True},
        baseline_overall=1.0,
        baseline_by_family={"iid": 1.0, "manifold": 1.0},
        n_seeds=1,
        n_draw=40,
        rng=np.random.default_rng(0),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="score a synthetic table and print the CSV columns")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--open-blind", action="store_true", help="read the sealed file after the dry run")
    args = parser.parse_args(argv)
    if not args.dry_run:
        raise SystemExit("train_monitor does not start training; pass --dry-run")
    out = args.out if args.out is not None else Path(tempfile.mkdtemp(prefix="train_monitor_"))
    frame = dry_run(out)
    print("training was not started")
    print("columns: " + ",".join(frame.columns))
    print(frame.to_csv(index=False), end="")
    try:
        read_sealed_blind(out / SEALED_NAME, open_blind=False)
    except BlindSealed as exc:
        print(exc)
    if args.open_blind:
        opened = read_sealed_blind(out / SEALED_NAME, open_blind=True)
        print(f"opened blind rows: {len(opened)}")
    else:
        print(f"sealed file left closed: {out / SEALED_NAME}")


if __name__ == "__main__":
    main()
