"""Shared paths and harness helpers for the Phase 1 audit.

Metrics come from ``score_harness``. This module does not reimplement log-L1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

AUDIT = Path(__file__).resolve().parent
ANALYSIS = AUDIT.parent
VARDOMAIN = ANALYSIS.parent
REPO = VARDOMAIN.parents[1]
CASE7A = REPO / "experiments" / "case7a"
for _path in (AUDIT, ANALYSIS, VARDOMAIN, CASE7A):
    text = str(_path)
    if text not in sys.path:
        sys.path.insert(0, text)

from score_harness import (  # noqa: E402
    BAND_NAMES,
    band_mask,
    bootstrap_independent,
    g_log_l1,
    git_commit,
    prior_log_l1,
    sample_id_hash,
    write_provenance,
)
from gates import NOISE_FLOOR_PLACEHOLDER_LOG_L1  # noqa: E402

CACHE = REPO / "data" / "vardomain" / "cache"
BOX = Path("/mnt/box/GIG Lab - UC Berkeley/Projects/Neural Operator/data")
CKPT = REPO / "checkpoints" / "vardomain" / "v2_ablations" / "P128n128"
PROPOSAL_FLOOR = float(NOISE_FLOOR_PLACEHOLDER_LOG_L1)
VS_REF = 400.0
ZETA_REF = 0.02
PCA_HEIGHT = 256


def use_agg() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)


def band_l1(diff: np.ndarray, freq: np.ndarray) -> dict[str, np.ndarray]:
    """Mean absolute error on each harness band. Matches ``g_log_l1`` / ``prior_log_l1``."""

    diff = np.asarray(diff, dtype=np.float64)
    out = {}
    for band in BAND_NAMES:
        mask = band_mask(freq, band)
        out[band] = np.mean(np.abs(diff[..., mask]), axis=-1)
    return out


def mean_pairwise_abs(curves: np.ndarray) -> np.ndarray:
    """Mean absolute difference over unordered pairs, one value per frequency.

    ``curves`` is (n, n_freq) with n >= 2. Sorting each frequency makes the
    pair average exact.
    """

    values = np.asarray(curves, dtype=np.float64)
    n = int(values.shape[0])
    if n < 2:
        raise ValueError("pairwise spread needs at least two curves")
    order = np.argsort(values, axis=0)
    ordered = np.take_along_axis(values, order, axis=0)
    rank = np.arange(1, n + 1, dtype=np.float64)[:, None]
    weight = 2.0 * rank - n - 1.0
    return (2.0 / (n * (n - 1))) * np.sum(weight * ordered, axis=0)


def provenance(path: Path, *, campaign: str, engine: str, outcrop: str, damping: str,
               fft, checkpoint, epoch, dtype: str, unit: str, sample_ids, difference: str,
               **extra) -> None:
    fields = {
        "label_campaign": campaign,
        "oned_engine": engine,
        "outcrop_convention": outcrop,
        "damping_source": damping,
        "fft_window_s": fft,
        "checkpoint": checkpoint,
        "epoch": epoch,
        "dtype": dtype,
        "bands": list(BAND_NAMES),
        "aggregation_unit": unit,
        "sample_id_hash": sample_id_hash(sample_ids),
        "git_commit": git_commit(),
        "definition_difference": difference,
    }
    fields.update(extra)
    write_provenance(path, fields)


def save_fig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    fig.clf()


def column_codes(vs: np.ndarray, zeta: np.ndarray, n_active: int, columns: np.ndarray,
                 mean: np.ndarray, components: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project observed columns onto the stored depth PCA.

    Features are interleaved ``log(Vs/400)`` and ``log(zeta/0.02)``, the inputs
    of ``FrozenColumnPCA``. Rows at or below ``n_active`` stay at the mean, which
    is how a short column is masked. ``depth_probe.physical_features`` is four
    scalars and is not this 512-wide vector.
    """

    height = int(components.shape[1] // 2)
    observed = min(int(n_active), height, int(vs.shape[0]))
    cols = np.asarray(columns, dtype=np.int64)
    n = int(cols.size)
    filled = np.repeat(np.asarray(mean, dtype=np.float64)[None, :], n, axis=0)
    if observed > 0:
        raw_vs = np.asarray(vs[:observed][:, cols], dtype=np.float64)
        raw_zeta = np.asarray(zeta[:observed][:, cols], dtype=np.float64)
        feat = np.empty((n, 2 * observed), dtype=np.float64)
        feat[:, 0::2] = np.log(np.maximum(raw_vs, 1e-6) / VS_REF).T
        feat[:, 1::2] = np.log(np.maximum(raw_zeta, 1e-8) / ZETA_REF).T
        filled[:, : 2 * observed] = feat
    else:
        feat = np.zeros((n, 0), dtype=np.float64)
    centered = filled - np.asarray(mean, dtype=np.float64)
    code = centered @ np.asarray(components, dtype=np.float64).T
    recon = code @ np.asarray(components, dtype=np.float64) + np.asarray(mean, dtype=np.float64)
    if observed > 0:
        err = np.mean(np.abs(recon[:, : 2 * observed] - feat), axis=1)
    else:
        err = np.zeros(n, dtype=np.float64)
    return code, err


def corr_length_m(row: np.ndarray) -> float:
    """Integral scale of a lateral log-Vs row, in metres at 1 m spacing."""

    z = np.asarray(row, dtype=np.float64)
    z = z - np.mean(z)
    width = int(z.size)
    if width < 4:
        return float("nan")
    spec = np.fft.rfft(z, n=2 * width)
    ac = np.fft.irfft(spec * np.conj(spec))[:width]
    if abs(ac[0]) < 1e-12:
        return float("nan")
    ac = ac / ac[0]
    stop = np.flatnonzero(ac[1:] <= 0.0)
    end = int(stop[0]) + 1 if stop.size else width
    return float(np.sum(ac[1:end]))


def self_check() -> None:
    """Harness identity and the pairwise reduction, on a toy curve."""

    freq = np.geomspace(0.1, 10.0, 1000)
    curve = np.sin(np.linspace(0.0, 8.0, 1000))
    for band in BAND_NAMES:
        got = float(np.mean(np.abs(curve[band_mask(freq, band)])))
        ref = prior_log_l1(curve, freq, band)
        if abs(got - ref) > 1e-12:
            raise AssertionError(f"{band} band_l1 {got} != prior_log_l1 {ref}")
        ref_g = g_log_l1(np.zeros_like(curve), curve, freq, band)
        if abs(got - ref_g) > 1e-12:
            raise AssertionError(f"{band} g_log_l1 mismatch")
    curves = np.stack([
        np.array([1.0, 0.0]),
        np.array([2.0, 0.0]),
        np.array([4.0, 5.0]),
    ])
    pair = mean_pairwise_abs(curves)
    # pairs (1,2)=1, (1,4)=3, (2,4)=2 -> 2; second column pairs 0,5,5 -> 10/3
    if abs(pair[0] - 2.0) > 1e-12 or abs(pair[1] - 10.0 / 3.0) > 1e-12:
        raise AssertionError(f"pairwise formula {pair}")
    boot = bootstrap_independent(
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array(["a", "a", "b", "c"]),
        unit="record",
        statistic=np.median,
        n_draw=20,
        rng=np.random.default_rng(0),
    )
    if boot["unit"] != "record":
        raise AssertionError(boot)
    try:
        bootstrap_independent(np.array([1.0]), np.array(["a"]), unit="station")
    except Exception as exc:  # AggregationError
        if type(exc).__name__ != "AggregationError":
            raise
    else:
        raise AssertionError("station bootstrap was accepted")
