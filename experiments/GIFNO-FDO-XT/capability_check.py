#!/usr/bin/env python3
"""
Surrogate vs OpenSees transfer functions on seiskit capability-check H5 cases.

TF space convention (must match training/eval):
  - Ground truth: linear |TF| from seiskit ``TTF_batch_fast`` (same as
    ``tf_per_sample.npy`` / ``compute_transfer_function.py``).
  - Model output: linear |TF| (optional softplus/exp via OUTPUT_ACTIVATION).
  - Headline metric ``rel_l2_mean`` is relative L2 on linear |TF| (same as
    ``test_rel_l2`` in ``eval_checkpoint.py``).
  - ``logspec_rel_l2_mean`` compares log(|TF|) on both sides (same as
    ``test_logspec_rel_l2_mean``).

Pipeline per case:
  1. Load cached ground-truth TF (or compute from accel + seiskit TTF, then cache)
  2. Build model input from Vs / zeta grids (optional Vs_macro / Vs_rf split)
  3. Forward pass with a saved GIFNO-FDO-XT checkpoint
  4. Compare prediction vs truth (metrics + plots)

Default H5 sources (override with --h5):
  ~/seiskit/neural-operator/experiments/three_layer/h5/case_*.h5  (3 cases)
  ~/seiskit/neural-operator/experiments/dipping/h5/case_*.h5      (2 cases)

Example:
  export GIFNO_MODEL_DIR=~/surrogate-seismic-waves/checkpoints/xt_lat128_d128
  export GIFNO_LATENT_CHANNELS=128 GIFNO_DEEPONET_LATENT_DIM=128

  cd experiments/GIFNO-FDO-XT-LOGLO-POD
  uv run python capability_check.py --all
  uv run python capability_check.py --h5 ~/seiskit/.../three_layer/h5/case_0.h5
  uv run python capability_check.py --all --force-gt   # recompute cached truth TFs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import hdf5plugin  # noqa: F401
except ImportError as e:
    raise ImportError(
        "hdf5plugin is required to read compressed capability-check H5 files."
    ) from e

import importlib.util

import config

config.setup_import_paths()

_GIFNO_CONFIG_PATH = Path(__file__).resolve().parents[1] / "GIFNO" / "config.py"
_spec = importlib.util.spec_from_file_location("gifno_config", _GIFNO_CONFIG_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load GIFNO config from {_GIFNO_CONFIG_PATH}")
_gifno_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gifno_config)

from data_loader import (  # noqa: E402
    _build_sample_coord_grids,
    _normalize_vs_by_surface,
    _normalize_zeta_by_max,
    _pad_depth,
)
from metrics import (  # noqa: E402
    pearson_1d,
    per_sample_logspec_rel_l2_numpy,
    per_sample_rel_l2_numpy,
)
from model import create_model  # noqa: E402

DEFAULT_SEISKIT_ROOT = Path.home() / "seiskit" / "neural-operator" / "experiments"
DEFAULT_OUT_ROOT = (
    Path.home() / "surrogate-seismic-waves" / "checkpoints" / "capability_checks_xt"
)
_EPS = 1e-12


def _ensure_seiskit_on_path() -> None:
    """Add seiskit to sys.path when not installed as a package."""
    if "seiskit" in sys.modules:
        return
    candidates = [
        os.environ.get("SEISKIT_ROOT"),
        str(Path.home() / "seiskit"),
    ]
    opensees_root = getattr(_gifno_config, "OPENSEES_ROOT", None)
    if opensees_root is not None:
        candidates.append(str(opensees_root))
    for root in candidates:
        if root and Path(root).is_dir() and root not in sys.path:
            sys.path.insert(0, root)


def default_capability_h5_paths(root: Path = DEFAULT_SEISKIT_ROOT) -> list[Path]:
    paths: list[Path] = []
    for exp in ("three_layer", "dipping"):
        h5_dir = root / exp / "h5"
        if h5_dir.is_dir():
            paths.extend(sorted(h5_dir.glob("case_*.h5")))
    return paths


def case_slug(h5_path: Path) -> str:
    """e.g. three_layer/case_0.h5 -> three_layer_case_0."""
    return f"{h5_path.parent.parent.name}_{h5_path.stem}"


def case_output_dir(out_root: Path, h5_path: Path) -> Path:
    return out_root / case_slug(h5_path)


def _gt_meta_path(case_dir: Path) -> Path:
    return case_dir / "gt_meta.json"


def _gt_cache_valid(h5_path: Path, case_dir: Path) -> bool:
    tf_path = case_dir / "tf_true.npy"
    freq_path = case_dir / "freq.npy"
    meta_path = _gt_meta_path(case_dir)
    if not (tf_path.is_file() and freq_path.is_file()):
        return False
    if not meta_path.is_file():
        return True
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return False
    return (
        meta.get("h5_path") == str(h5_path.resolve())
        and meta.get("h5_mtime") == h5_path.stat().st_mtime
        and meta.get("h5_size") == h5_path.stat().st_size
    )


def compute_ground_truth_tf(h5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """OpenSees accel -> (n_recorders, n_freq) via seiskit TTF_batch_fast."""
    _ensure_seiskit_on_path()
    from seiskit.ttf.TTF import TTF_batch_fast  # noqa: WPS433

    n_lateral = config.N_LATERAL
    with h5py.File(h5_path, "r") as f:
        data = f["recorders/accel/data"][:]
        dt = float(f["grid"].attrs["dt"])

    if data.shape[1] != 2 * n_lateral:
        raise ValueError(
            f"Expected {2 * n_lateral} accel channels in {h5_path}, got {data.shape[1]}"
        )

    base_2d = data[:, :n_lateral].T
    surf_2d = data[:, n_lateral:].T
    freq, mags = TTF_batch_fast(
        base_2d,
        surf_2d,
        dt=dt,
        dz=config.DZ,
        smooth_coeff=_gifno_config.SMOOTH_COEFF,
        Vsmin=None,
        n_points=config.N_FREQ,
    )
    if len(freq) != config.N_FREQ:
        raise RuntimeError(f"Unexpected n_freq={len(freq)} for {h5_path.name}")
    return mags.astype(np.float32), freq.astype(np.float32)


def load_or_compute_ground_truth(
    h5_path: Path,
    case_dir: Path,
    *,
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    case_dir.mkdir(parents=True, exist_ok=True)
    tf_path = case_dir / "tf_true.npy"
    freq_path = case_dir / "freq.npy"

    if not force and _gt_cache_valid(h5_path, case_dir):
        print(f"[capability] Using cached ground truth: {case_dir}")
        return np.load(tf_path), np.load(freq_path)

    print(f"[capability] Computing ground truth TF from {h5_path.name} ...")
    tf_true, freq = compute_ground_truth_tf(h5_path)
    np.save(tf_path, tf_true)
    np.save(freq_path, freq)
    np.save(case_dir / "recorder_x.npy", config.recorder_x_indices())
    meta = {
        "h5_path": str(h5_path.resolve()),
        "h5_mtime": h5_path.stat().st_mtime,
        "h5_size": h5_path.stat().st_size,
        "n_recorders": int(tf_true.shape[0]),
        "n_freq": int(tf_true.shape[1]),
    }
    _gt_meta_path(case_dir).write_text(json.dumps(meta, indent=2))
    print(f"[capability] Cached ground truth -> {case_dir}")
    return tf_true, freq


def build_input_from_h5(h5_path: Path) -> np.ndarray:
    """H5 Vs/zeta grids -> (C, NZ_MAX, NX) float32 model input."""
    with h5py.File(h5_path, "r") as f:
        vs_raw = f["Vs_realization_2D"][:]
        zeta_raw = f["Damping_zeta"][:]
        nz = int(vs_raw.shape[0])
        lz = float(f["grid"].attrs["Lz"])
        dz = float(f["grid"].attrs.get("dz", config.DZ))
        dx = float(f["grid"].attrs.get("dx", config.DX))

        sl = slice(config.X_SLICE_START, config.X_SLICE_END)
        vs = _pad_depth(vs_raw, config.NZ_MAX)[:, sl]
        zeta = _pad_depth(zeta_raw, config.NZ_MAX)[:, sl]
        x_coord, z_coord = _build_sample_coord_grids(
            nz,
            config.NX,
            config.NZ_MAX,
            lz,
            float(config.LX_VARIABILITY),
            dz=dz,
            dx=dx,
        )

    if vs[0].max() <= 0:
        raise ValueError(f"Surface row has zero Vs in {h5_path}")

    vs = _normalize_vs_by_surface(vs, config.VS_NORM_EPS)
    zeta = _normalize_zeta_by_max(zeta, nz, config.ZETA_NORM_EPS)
    from data_loader import stack_model_input_channels

    return stack_model_input_channels(vs, zeta, x_coord, z_coord).astype(np.float32)


def load_model(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = create_model().to(device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    model.eval()
    return model


@torch.no_grad()
def predict_tf(
    model: torch.nn.Module,
    x: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Forward pass -> (n_recorders, n_freq)."""
    grid = model(torch.from_numpy(x[None]).to(device)).cpu().numpy()[0]
    rec = config.recorder_x_indices()
    return grid[rec, :].astype(np.float32)


def _band_mask(freq: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    lo, hi = band
    return (freq >= lo) & (freq <= hi)


def _rel_l2(pred: np.ndarray, true: np.ndarray) -> float:
    num = np.linalg.norm(pred - true)
    den = np.linalg.norm(true) + _EPS
    return float(num / den)


def _check_pod_basis_space() -> None:
    """Warn when POD mean looks inconsistent with linear-amplitude training TFs."""
    pod_mean_path = getattr(config, "POD_MEAN_PATH", None)
    if pod_mean_path is None or not Path(pod_mean_path).is_file():
        return
    mean = np.load(pod_mean_path)
    if float(mean.min()) < -1.0:
        print(
            "[capability] WARNING: pod_mean has large negatives "
            f"(min={mean.min():.3f}). Training TFs are linear |TF| (positive); "
            "POD basis may be mismatched with best_model.pt. "
            "Re-sync pod_mean.npy and pod_modes.npy from the sweep checkpoint."
        )


def compare_tfs(
    tf_pred: np.ndarray,
    tf_true: np.ndarray,
    freq: np.ndarray,
    *,
    case_name: str,
    out_dir: Path,
) -> dict[str, Any]:
    """Metrics + diagnostic plots for one capability case (linear |TF| space)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = config.recorder_x_indices()
    n_rec = len(rec)

    # Match eval_checkpoint / metrics.py conventions.
    pred_grid = np.zeros((1, config.NX, config.N_FREQ), dtype=np.float32)
    true_grid = np.zeros_like(pred_grid)
    for ch, x_idx in enumerate(rec):
        pred_grid[0, x_idx, :] = tf_pred[ch]
        true_grid[0, x_idx, :] = tf_true[ch]

    rel_l2_sample = float(per_sample_rel_l2_numpy(pred_grid, true_grid, rec)[0])
    logspec_rel_l2_sample = float(
        per_sample_logspec_rel_l2_numpy(pred_grid, true_grid, rec)[0]
    )

    rel_l2 = np.array(
        [_rel_l2(tf_pred[r], tf_true[r]) for r in range(n_rec)], dtype=np.float64
    )
    logspec_rel_l2 = np.array(
        [
            _rel_l2(
                np.log(np.maximum(tf_pred[r], _EPS)),
                np.log(np.maximum(tf_true[r], _EPS)),
            )
            for r in range(n_rec)
        ],
        dtype=np.float64,
    )
    pearson = np.array(
        [pearson_1d(tf_pred[r], tf_true[r]) for r in range(n_rec)], dtype=np.float64
    )

    band_metrics: dict[str, float] = {}
    for name, band in (
        ("low", config.FREQ_BAND_LOW),
        ("mid", config.FREQ_BAND_MID),
        ("high", config.FREQ_BAND_HIGH),
    ):
        mask = _band_mask(freq, band)
        if not np.any(mask):
            continue
        band_metrics[f"rel_l2_band_{name}_mean"] = float(
            np.mean([_rel_l2(tf_pred[r, mask], tf_true[r, mask]) for r in range(n_rec)])
        )
        band_metrics[f"logspec_rel_l2_band_{name}_mean"] = float(
            np.mean(
                [
                    _rel_l2(
                        np.log(np.maximum(tf_pred[r, mask], _EPS)),
                        np.log(np.maximum(tf_true[r, mask], _EPS)),
                    )
                    for r in range(n_rec)
                ]
            )
        )

    central_r = int(np.argmin(np.abs(rec - config.NX // 2)))
    worst_r = int(np.argmax(rel_l2))

    metrics: dict[str, Any] = {
        "case": case_name,
        "n_recorders": n_rec,
        "tf_space": "linear_amplitude",
        "rel_l2_mean": float(np.mean(rel_l2)),
        "rel_l2_median": float(np.median(rel_l2)),
        "rel_l2_p10": float(np.percentile(rel_l2, 10)),
        "rel_l2_p90": float(np.percentile(rel_l2, 90)),
        "rel_l2_max": float(np.max(rel_l2)),
        "rel_l2_sample": rel_l2_sample,
        "logspec_rel_l2_mean": float(np.mean(logspec_rel_l2)),
        "logspec_rel_l2_median": float(np.median(logspec_rel_l2)),
        "logspec_rel_l2_sample": logspec_rel_l2_sample,
        "pearson_mean": float(np.nanmean(pearson)),
        "pearson_median": float(np.nanmedian(pearson)),
        "rel_l2_central": float(rel_l2[central_r]),
        "logspec_rel_l2_central": float(logspec_rel_l2[central_r]),
        "pearson_central": float(pearson[central_r]),
        "rel_l2_worst_recorder": float(rel_l2[worst_r]),
        "logspec_rel_l2_worst_recorder": float(logspec_rel_l2[worst_r]),
        "pearson_worst_recorder": float(pearson[worst_r]),
        "worst_recorder_x": int(rec[worst_r]),
        "central_recorder_x": int(rec[central_r]),
        **band_metrics,
        "per_recorder_rel_l2": rel_l2.tolist(),
        "per_recorder_logspec_rel_l2": logspec_rel_l2.tolist(),
        "per_recorder_pearson": pearson.tolist(),
        "recorder_x": rec.tolist(),
    }

    # --- plots ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogx(freq, tf_true[central_r], label="OpenSees (truth)", lw=1.5)
    ax.semilogx(freq, tf_pred[central_r], "--", label="surrogate", lw=1.5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Linear |TF|")
    ax.set_title(
        f"{case_name} — central recorder x={rec[central_r]} m\n"
        f"rel_l2={rel_l2[central_r]:.3f}, "
        f"logspec={logspec_rel_l2[central_r]:.3f}, r={pearson[central_r]:.3f}"
    )
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_central.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogx(freq, tf_true[worst_r], label="OpenSees (truth)", lw=1.5)
    ax.semilogx(freq, tf_pred[worst_r], "--", label="surrogate", lw=1.5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Linear |TF|")
    ax.set_title(
        f"{case_name} — worst recorder x={rec[worst_r]} m\n"
        f"rel_l2={rel_l2[worst_r]:.3f}, "
        f"logspec={logspec_rel_l2[worst_r]:.3f}, r={pearson[worst_r]:.3f}"
    )
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_worst.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    x_m = (rec.astype(float) - config.NX // 2) * config.DX
    ax.plot(x_m, rel_l2, "o-", label="rel L2")
    ax.set_xlabel("Lateral offset from center (m)")
    ax.set_ylabel("Relative L2")
    ax.set_title(f"{case_name} — per-recorder error")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "rel_l2_by_recorder.png", dpi=150)
    plt.close(fig)

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    return metrics


def run_case(
    h5_path: Path,
    *,
    out_root: Path,
    checkpoint: Path,
    device: torch.device,
    force_gt: bool = False,
) -> dict[str, Any]:
    h5_path = h5_path.resolve()
    if not h5_path.is_file():
        raise FileNotFoundError(h5_path)

    slug = case_slug(h5_path)
    case_dir = case_output_dir(out_root, h5_path)
    print(f"\n=== {slug} ===")

    tf_true, freq = load_or_compute_ground_truth(h5_path, case_dir, force=force_gt)

    print("[capability] Building model input ...")
    x = build_input_from_h5(h5_path)

    print(f"[capability] Loading checkpoint {checkpoint}")
    model = load_model(checkpoint, device)

    print("[capability] Forward pass ...")
    tf_pred = predict_tf(model, x, device)
    np.save(case_dir / "tf_pred.npy", tf_pred)

    print("[capability] Comparing prediction vs truth ...")
    metrics = compare_tfs(tf_pred, tf_true, freq, case_name=slug, out_dir=case_dir)
    print(
        f"  rel_l2_mean={metrics['rel_l2_mean']:.4f}  "
        f"logspec_rel_l2_mean={metrics['logspec_rel_l2_mean']:.4f}  "
        f"pearson_mean={metrics['pearson_mean']:.4f}"
    )
    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GIFNO-FDO-XT surrogate vs OpenSees TF on capability-check H5 cases"
    )
    parser.add_argument(
        "--h5",
        type=Path,
        nargs="*",
        default=None,
        help="One or more capability-check H5 paths (default: --all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=f"Run all cases under {DEFAULT_SEISKIT_ROOT}",
    )
    parser.add_argument(
        "--seiskit-root",
        type=Path,
        default=DEFAULT_SEISKIT_ROOT,
        help="Root of seiskit neural-operator experiments",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="Output root for cached truth TFs, predictions, and plots",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to best_model.pt (default: config.MODEL_SAVE_PATH)",
    )
    parser.add_argument(
        "--force-gt",
        action="store_true",
        help="Recompute ground-truth TF even when a valid cache exists",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cpu or cuda (default: config.DEVICE)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    checkpoint = Path(args.checkpoint or config.MODEL_SAVE_PATH)
    if not checkpoint.is_file():
        print(f"ERROR: checkpoint not found: {checkpoint}", file=sys.stderr)
        return 1

    if args.h5:
        h5_paths = [Path(p) for p in args.h5]
    elif args.all:
        h5_paths = default_capability_h5_paths(args.seiskit_root)
    else:
        h5_paths = default_capability_h5_paths(args.seiskit_root)
        if not h5_paths:
            print(
                "No capability H5 files found. Pass --h5 PATH ... or --all with a "
                f"valid --seiskit-root (tried {args.seiskit_root}).",
                file=sys.stderr,
            )
            return 1

    if not h5_paths:
        print("ERROR: no H5 files to process.", file=sys.stderr)
        return 1

    device = torch.device(args.device if args.device else str(config.DEVICE))

    all_metrics: list[dict[str, Any]] = []
    for h5_path in h5_paths:
        all_metrics.append(
            run_case(
                h5_path,
                out_root=args.out_dir,
                checkpoint=checkpoint,
                device=device,
                force_gt=args.force_gt,
            )
        )

    summary_path = args.out_dir / "summary.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_metrics, indent=2))
    print(f"\n[capability] Wrote summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
