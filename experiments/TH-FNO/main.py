#!/usr/bin/env python3
"""TH-FNO entrypoint — train on GIFNO corpus (requires GIFNO_DATA_ROOT)."""

from __future__ import annotations

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

import argparse
import json

import wandb

import config

config.setup_import_paths()

from gifno_dataset import get_gifno_loaders  # noqa: E402
from tf_plots import log_tf_comparison_plots  # noqa: E402
from train import evaluate_loader, train_model  # noqa: E402


def _set_seed(seed: int) -> None:
    import random

    import numpy as np
    import torch

    config.SEED = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override config.SEED (data split + init). Required for seeded A/B.",
    )
    p.add_argument(
        "--predict-mode",
        choices=("direct", "residual"),
        default=None,
        help="Override THFNO_PREDICT_MODE / config.PREDICT_MODE",
    )
    args = p.parse_args()

    if args.seed is not None:
        _set_seed(args.seed)
    else:
        _set_seed(config.SEED)

    if args.predict_mode is not None:
        config.PREDICT_MODE = args.predict_mode
        if "direct" in config.WANDB_RUN_NAME or "residual" in config.WANDB_RUN_NAME:
            pass
        else:
            config.WANDB_RUN_NAME = f"{config.WANDB_RUN_NAME}_{args.predict_mode}"

    mode = config.PREDICT_MODE
    seed_tag = f"_s{config.SEED}" if args.seed is not None else ""
    config.MODEL_SAVE_DIR = (
        config.MODEL_SAVE_DIR.parent / f"th_fno_{mode}{seed_tag}"
    )
    config.MODEL_SAVE_PATH = config.MODEL_SAVE_DIR / "best_model.pt"
    config.MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = get_gifno_loaders(
        limit=args.limit, batch_size=args.batch_size
    )
    print(
        f"[TH-FNO] train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
        f"test={len(test_loader.dataset)} device={config.DEVICE} "
        f"PREDICT_MODE={config.PREDICT_MODE} RESIDUAL_MODE={config.RESIDUAL_MODE} "
        f"AMP_DOMAIN={config.AMPLITUDE_DOMAIN} "
        f"TREND_FREQ_SCALE={config.TREND_FREQ_SCALE} "
        f"ckpt={config.MODEL_SAVE_PATH}"
    )
    if config.AMPLITUDE_DOMAIN != "linear":
        print(
            f"[TH-FNO] WARNING: AMPLITUDE_DOMAIN={config.AMPLITUDE_DOMAIN} "
            "(prefer linear / raw |TF| for this experiment)",
            flush=True,
        )
    if config.PREDICT_MODE == "residual":
        go = config.DIAGNOSTICS_DIR / "d2_pass.json"
        if go.is_file():
            d2 = json.loads(go.read_text())
            if not d2.get("pass", False):
                print(
                    "[TH-FNO] NOTE: residual A/B run despite D2 FAIL "
                    "(intentional diagnostics check)."
                )
    model = train_model(train_loader, val_loader, num_epochs=args.epochs)
    metrics = evaluate_loader(model, test_loader, config.DEVICE)
    metrics["predict_mode"] = config.PREDICT_MODE
    metrics["seed"] = int(config.SEED)
    metrics["residual_mode"] = config.RESIDUAL_MODE
    metrics["trend_freq_scale"] = float(config.TREND_FREQ_SCALE)
    metrics["log_delta_c"] = float(getattr(config, "LOG_DELTA_C", config.LOG_DELTA_CLAMP))
    metrics["loss_term_norm"] = bool(config.LOSS_TERM_NORM)
    metrics["zero_init_residual_head"] = bool(
        getattr(config, "ZERO_INIT_RESIDUAL_HEAD", True)
    )
    print("[TH-FNO] test:", metrics)
    out_name = f"gifno_test_metrics_{mode}_s{config.SEED}.json"
    out = config.RESULTS_SAVE_DIR / out_name
    out.write_text(json.dumps(metrics, indent=2))
    # Also write unseeded alias for single-run convenience
    (config.RESULTS_SAVE_DIR / f"gifno_test_metrics_{mode}.json").write_text(
        json.dumps(metrics, indent=2)
    )
    print(f"Wrote {out}")

    # GIFNO-style center + edge |TF|(f) overlays → W&B Media
    try:
        plot_stats = log_tf_comparison_plots(
            model,
            test_loader,
            config.DEVICE,
            tag="test",
            max_collect=getattr(config, "EVAL_TF_COLLECT", 64),
        )
        metrics.update(plot_stats)
        if wandb.run is not None:
            wandb.run.summary.update(metrics)
    except Exception as e:
        print(f"[TH-FNO] TF plot logging failed: {e}", flush=True)
    finally:
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
