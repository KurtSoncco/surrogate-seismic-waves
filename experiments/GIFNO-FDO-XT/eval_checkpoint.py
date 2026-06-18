#!/usr/bin/env python3
"""
Evaluate a saved GIFNO-FDO-XT checkpoint on the test split and write analysis artifacts.

Default output: experiments/GIFNO-FDO-XT/analysis/<run_name>/

Example:
  export GIFNO_DATA_ROOT="/mnt/box_lab/Projects/Neural Operator/data"
  export GIFNO_LATENT_CHANNELS=128
  export GIFNO_DEEPONET_LATENT_DIM=128

  python eval_checkpoint.py \\
    --checkpoint ~/surrogate-seismic-waves/checkpoints/xt_lat128_d128/best_model.pt \\
    --run-name xt_lat128_d128
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import config

config.setup_import_paths()

from data_loader import get_data_loaders  # noqa: E402
from evaluate import evaluate_model  # noqa: E402
from metrics import (  # noqa: E402
    aggregate_test_metrics,
    per_recorder_pearson_numpy,
    per_recorder_rel_l2_numpy,
)
from model import create_model  # noqa: E402

_ANALYSIS_ROOT = Path(__file__).resolve().parent / "analysis"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate XT checkpoint on test set")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to best_model.pt (default: config.MODEL_SAVE_PATH)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="xt_lat128_d128",
        help="Subfolder under experiments/GIFNO-FDO-XT/analysis/",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override output directory (default: analysis/<run_name>)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Test DataLoader batch size (lower if RAM is tight)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader workers",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate PNG diagnostic plots (uses more RAM)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda or cpu (default: auto)",
    )
    return parser.parse_args()


def _run_inference(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    recorder_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return recorder-gathered pred/target (N, R, F) and masks (N, Nx)."""
    n_test = len(test_loader.dataset)
    n_rec = len(recorder_x)
    n_freq = config.N_FREQ
    nx = config.NX

    pred_rec = np.lib.format.open_memmap(
        "_pred_rec_tmp.npy", mode="w+", dtype=np.float32, shape=(n_test, n_rec, n_freq)
    )
    tgt_rec = np.lib.format.open_memmap(
        "_tgt_rec_tmp.npy", mode="w+", dtype=np.float32, shape=(n_test, n_rec, n_freq)
    )
    masks = np.lib.format.open_memmap(
        "_masks_tmp.npy", mode="w+", dtype=np.float32, shape=(n_test, nx)
    )

    idx_cpu = torch.as_tensor(recorder_x, dtype=torch.long)
    offset = 0
    model.eval()
    with torch.no_grad():
        for inputs, targets, batch_masks in tqdm(test_loader, desc="Inference"):
            b = inputs.shape[0]
            inputs = inputs.to(device)
            outputs = model(inputs)
            idx = idx_cpu.to(outputs.device)
            out_rec = outputs.index_select(1, idx).cpu().numpy()
            tgt_rec_batch = targets.index_select(1, idx_cpu).numpy()
            pred_rec[offset : offset + b] = out_rec
            tgt_rec[offset : offset + b] = tgt_rec_batch
            masks[offset : offset + b] = batch_masks.numpy()
            offset += b

    pred_arr = np.array(pred_rec)
    tgt_arr = np.array(tgt_rec)
    mask_arr = np.array(masks)
    for p in ("_pred_rec_tmp.npy", "_tgt_rec_tmp.npy", "_masks_tmp.npy"):
        Path(p).unlink(missing_ok=True)
    return pred_arr, tgt_arr, mask_arr


def main() -> int:
    args = _parse_args()
    out_dir = args.out_dir or (_ANALYSIS_ROOT / args.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = args.checkpoint or config.MODEL_SAVE_PATH
    ckpt = Path(ckpt)
    if not ckpt.is_file():
        print(f"ERROR: checkpoint not found: {ckpt}", file=sys.stderr)
        return 1

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    _, _, test_loader, freq = get_data_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = create_model().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))

    recorder_x = config.recorder_x_indices()
    pred_rec, tgt_rec, masks = _run_inference(model, test_loader, device, recorder_x)

    # Full-grid scatter for evaluate_model / metrics (zeros off-recorder)
    nx, n_freq = config.NX, config.N_FREQ
    n_test = pred_rec.shape[0]
    predictions = np.zeros((n_test, nx, n_freq), dtype=np.float32)
    targets = np.zeros((n_test, nx, n_freq), dtype=np.float32)
    predictions[:, recorder_x, :] = pred_rec
    targets[:, recorder_x, :] = tgt_rec

    metrics, per_sample = aggregate_test_metrics(predictions, targets, masks, freq)
    rec_rel_l2 = per_recorder_rel_l2_numpy(predictions, targets, recorder_x)
    rec_pearson = per_recorder_pearson_numpy(predictions, targets, recorder_x)

    np.savez_compressed(
        out_dir / "test_recorder_tf.npz",
        predictions_rec=pred_rec,
        targets_rec=tgt_rec,
        masks=masks,
        recorder_x=recorder_x,
        freq=freq,
    )
    np.savez_compressed(
        out_dir / "per_sample_metrics.npz",
        **{k: np.asarray(v) for k, v in per_sample.items()},
    )
    np.savez_compressed(
        out_dir / "per_recorder_metrics.npz",
        rel_l2=rec_rel_l2,
        pearson=rec_pearson,
        recorder_x=recorder_x,
        x_norm=config.recorder_x_trunk_coords(recorder_x),
    )

    meta = {
        "checkpoint": str(ckpt.resolve()),
        "run_name": args.run_name,
        "n_test": int(n_test),
        "latent_channels": config.LATENT_CHANNELS,
        "deeponet_latent_dim": config.DEEPONET_LATENT_DIM,
        "branch_mode": config.BRANCH_MODE,
        "x_coord_mode": config.X_COORD_MODE,
        "loss_p": config.LOSS_P,
        "seed": config.SEED,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    (out_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"Saved analysis artifacts to {out_dir}")
    print("Test metrics:", metrics)

    if args.plots:
        plots_dir = out_dir / "plots"
        evaluate_model(
            model,
            test_loader,
            freq_data=freq,
            save_dir=plots_dir,
            run=None,
            seed=config.SEED,
        )
        print(f"Plots saved to {plots_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
