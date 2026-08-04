"""Compare Conv vs ResUNet field encoders on signed R_nom.

Trains ResUNet if missing, evaluates both on the same test split, writes:
  - aggregate JSON
  - TF prediction overlay curves
  - per-sample metric histograms / boxplots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from data import ResidualDeepONetDataset, make_splits, stoch_dim, trunk_feature_names
from model import FieldEncoderKind, build_model
from train import (
    _build_signed_cache,
    _device,
    _forward,
    _pearson,
    _pearson_across_freq,
    _r2,
    _rel_l2,
    fit_and_apply_norms,
    train_one,
)

CENTRAL_REC = config.N_LATERAL // 2


def _load_model(
    ckpt_path: Path,
    *,
    field_encoder: FieldEncoderKind,
    trunk_set: str = "full",
) -> tuple[torch.nn.Module, Dict[str, Any]]:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    trunk_dim = len(trunk_feature_names(trunk_set))  # type: ignore[arg-type]
    model = build_model(
        blob.get("branch_mode", "single"),
        field_channels=config.FIELD_CHANNELS,
        stoch_dim=stoch_dim(),
        trunk_dim=trunk_dim,
        latent_dim=config.LATENT_DIM,
        field_hidden=config.FIELD_HIDDEN,
        branch_hidden=config.BRANCH_HIDDEN,
        trunk_hidden=config.TRUNK_HIDDEN,
        trunk_layers=config.TRUNK_LAYERS,
        field_encoder=field_encoder,
    )
    model.load_state_dict(blob["model"])
    model.eval()
    return model, blob


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    stats: Dict[str, torch.Tensor],
    device: torch.device,
    *,
    n_rec: int,
    n_freq: int,
    branch_mode: str = "single",
) -> Dict[str, np.ndarray]:
    """Return arrays shaped for per-sample and curve analysis."""
    t_mean = stats["target_mean"].to(device)
    t_std = stats["target_std"].to(device)
    R_true, R_pred, TF1D, TF2D, TF_hat = [], [], [], [], []
    for batch in loader:
        fields = batch["fields"].to(device)
        stoch = batch["stoch"].to(device)
        trunk_y = batch["trunk_y"].to(device)
        target = batch["target_raw"].to(device)
        tf1d = batch["tf1d"].to(device)
        tf2d = batch["tf2d"].to(device)
        pred_n = _forward(model, fields, stoch, trunk_y, branch_mode)  # type: ignore[arg-type]
        pred = pred_n * t_std + t_mean
        R_true.append(target.cpu().numpy())
        R_pred.append(pred.cpu().numpy())
        TF1D.append(tf1d.cpu().numpy())
        TF2D.append(tf2d.cpu().numpy())
        TF_hat.append((tf1d + pred).cpu().numpy())

    def _cat(xs: List[np.ndarray]) -> np.ndarray:
        return np.concatenate(xs, axis=0)

    rt, rp = _cat(R_true), _cat(R_pred)
    t1, t2, th = _cat(TF1D), _cat(TF2D), _cat(TF_hat)
    n_s = rt.shape[0]
    return {
        "R_true": rt.reshape(n_s, n_rec, n_freq),
        "R_pred": rp.reshape(n_s, n_rec, n_freq),
        "TF1D": t1.reshape(n_s, n_rec, n_freq),
        "TF2D": t2.reshape(n_s, n_rec, n_freq),
        "TF_hat": th.reshape(n_s, n_rec, n_freq),
    }


def per_sample_metrics(pack: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    n_s, n_rec, n_freq = pack["R_true"].shape
    r2_R, pearson_R_f, delta_tf, pearson_tf_f = [], [], [], []
    for i in range(n_s):
        y = pack["R_true"][i].ravel()
        p = pack["R_pred"][i].ravel()
        r2_R.append(_r2(y, p))
        pearson_R_f.append(
            _pearson_across_freq(y, p, n_rec=n_rec, n_freq=n_freq)
        )
        tf2d = pack["TF2D"][i].ravel()
        tf1d = pack["TF1D"][i].ravel()
        tfhat = pack["TF_hat"][i].ravel()
        delta_tf.append(_r2(tf2d, tfhat) - _r2(tf2d, tf1d))
        pearson_tf_f.append(
            _pearson_across_freq(tf2d, tfhat, n_rec=n_rec, n_freq=n_freq)
        )
    return {
        "r2_R": np.asarray(r2_R),
        "pearson_R_freq": np.asarray(pearson_R_f),
        "delta_r2_TF": np.asarray(delta_tf),
        "pearson_TF_freq": np.asarray(pearson_tf_f),
        "rel_l2_R": np.asarray(
            [
                _rel_l2(pack["R_true"][i].ravel(), pack["R_pred"][i].ravel())
                for i in range(n_s)
            ]
        ),
    }


def aggregate(pack: Dict[str, np.ndarray]) -> Dict[str, float]:
    n_rec = pack["R_true"].shape[1]
    n_freq = pack["R_true"].shape[2]
    y = pack["R_true"].ravel()
    p = pack["R_pred"].ravel()
    tf2d = pack["TF2D"].ravel()
    tf1d = pack["TF1D"].ravel()
    tfhat = pack["TF_hat"].ravel()
    return {
        "r2_R": _r2(y, p),
        "pearson_R": _pearson(y, p),
        "pearson_R_freq": _pearson_across_freq(y, p, n_rec=n_rec, n_freq=n_freq),
        "rel_l2_R": _rel_l2(y, p),
        "r2_TF": _r2(tf2d, tfhat),
        "r2_TF_1d_only": _r2(tf2d, tf1d),
        "delta_r2_TF": _r2(tf2d, tfhat) - _r2(tf2d, tf1d),
        "pearson_TF_freq": _pearson_across_freq(
            tf2d, tfhat, n_rec=n_rec, n_freq=n_freq
        ),
    }


def plot_metric_boxes(
    metrics: Dict[str, Dict[str, np.ndarray]],
    out_path: Path,
) -> None:
    keys = ["r2_R", "pearson_R_freq", "delta_r2_TF", "pearson_TF_freq"]
    titles = ["R² (signed R)", "Pearson R (freq)", "ΔR² TF vs TF₁D", "Pearson TF (freq)"]
    labels = list(metrics.keys())
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.6))
    for ax, key, title in zip(axes, keys, titles):
        data = [metrics[lab][key] for lab in labels]
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
        colors = ["#4C78A8", "#F58518"]
        for patch, c in zip(bp["boxes"], colors[: len(bp["boxes"])]):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax.set_title(title, fontsize=10)
        ax.axhline(0.0, color="k", lw=0.4)
        ax.tick_params(axis="x", labelsize=9)
    fig.suptitle("R_nom test set — per-sample metrics (Conv vs ResUNet)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metric_hists(
    metrics: Dict[str, Dict[str, np.ndarray]],
    out_path: Path,
) -> None:
    keys = ["r2_R", "pearson_R_freq", "delta_r2_TF"]
    titles = ["R² (signed R)", "Pearson R across freq", "ΔR² TF vs TF₁D-only"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, key, title in zip(axes, keys, titles):
        for lab, color in zip(metrics.keys(), ["#4C78A8", "#F58518"]):
            ax.hist(
                metrics[lab][key],
                bins=20,
                alpha=0.55,
                label=lab,
                color=color,
                density=True,
            )
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("R_nom — metric distributions on held-out samples", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_tf_curves(
    packs: Dict[str, Dict[str, np.ndarray]],
    freq: np.ndarray,
    out_path: Path,
    *,
    n_show: int = 4,
    seed: int = 42,
) -> None:
    rng = np.random.default_rng(seed)
    n_s = next(iter(packs.values()))["TF2D"].shape[0]
    idxs = rng.choice(n_s, size=min(n_show, n_s), replace=False)
    labels = list(packs.keys())
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    axes = axes.ravel()
    for ax, i in zip(axes, idxs):
        tf2d = packs[labels[0]]["TF2D"][i, CENTRAL_REC]
        tf1d = packs[labels[0]]["TF1D"][i, CENTRAL_REC]
        ax.plot(freq, tf2d, "k-", lw=1.6, label="OpenSees TF₂D")
        ax.plot(freq, tf1d, color="0.55", ls="--", lw=1.2, label="TF₁D nom")
        colors = ["#4C78A8", "#F58518"]
        for lab, c in zip(labels, colors):
            ax.plot(
                freq,
                packs[lab]["TF_hat"][i, CENTRAL_REC],
                color=c,
                lw=1.3,
                label=f"{lab} TF₁D+R̂",
            )
        ax.set_xscale("log")
        ax.set_title(f"test sample #{i} · central recorder", fontsize=9)
        ax.set_ylabel("|TF|")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=7, loc="best")
    axes[-2].set_xlabel("f [Hz]")
    axes[-1].set_xlabel("f [Hz]")
    fig.suptitle("R_nom TF reconstruction — Conv vs ResUNet branch encoder", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_residual_curves(
    packs: Dict[str, Dict[str, np.ndarray]],
    freq: np.ndarray,
    out_path: Path,
    *,
    n_show: int = 4,
    seed: int = 42,
) -> None:
    rng = np.random.default_rng(seed)
    n_s = next(iter(packs.values()))["R_true"].shape[0]
    idxs = rng.choice(n_s, size=min(n_show, n_s), replace=False)
    labels = list(packs.keys())
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    axes = axes.ravel()
    for ax, i in zip(axes, idxs):
        ax.plot(
            freq,
            packs[labels[0]]["R_true"][i, CENTRAL_REC],
            "k-",
            lw=1.5,
            label="true R_nom",
        )
        colors = ["#4C78A8", "#F58518"]
        for lab, c in zip(labels, colors):
            ax.plot(
                freq,
                packs[lab]["R_pred"][i, CENTRAL_REC],
                color=c,
                lw=1.2,
                label=f"{lab} R̂",
            )
        ax.axhline(0.0, color="k", lw=0.4)
        ax.set_xscale("log")
        ax.set_title(f"test sample #{i} · central recorder", fontsize=9)
        ax.set_ylabel("signed R")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=7)
    axes[-2].set_xlabel("f [Hz]")
    axes[-1].set_xlabel("f [Hz]")
    fig.suptitle("Signed R_nom spectra — Conv vs ResUNet", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-tag", default="n1000_seed42")
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--patience", type=int, default=config.PATIENCE)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument(
        "--skip-train",
        action="store_true",
        help="Only evaluate existing checkpoints",
    )
    args = p.parse_args()

    target = "R_nom"
    trunk_set = "full"
    cache_dir = _build_signed_cache(args.cache_tag)

    conv_name = f"single_{trunk_set}_{target}_{args.cache_tag}"
    res_name = f"single_resunet_{trunk_set}_{target}_{args.cache_tag}"
    conv_ckpt = config.CHECKPOINT_DIR / f"{conv_name}.pt"
    res_ckpt = config.CHECKPOINT_DIR / f"{res_name}.pt"

    if not args.skip_train:
        if not conv_ckpt.exists():
            print("Training Conv encoder baseline (R_nom)...", flush=True)
            train_one(
                cache_tag=args.cache_tag,
                target=target,  # type: ignore[arg-type]
                branch_mode="single",
                trunk_set=trunk_set,  # type: ignore[arg-type]
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.seed,
                patience=args.patience,
                field_encoder="conv",
            )
        else:
            print(f"Reusing Conv checkpoint {conv_ckpt}", flush=True)
        print("Training ResUNet encoder (R_nom)...", flush=True)
        train_one(
            cache_tag=args.cache_tag,
            target=target,  # type: ignore[arg-type]
            branch_mode="single",
            trunk_set=trunk_set,  # type: ignore[arg-type]
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            patience=args.patience,
            field_encoder="resunet",
        )

    if not conv_ckpt.exists() or not res_ckpt.exists():
        raise SystemExit(f"Missing checkpoints:\n  {conv_ckpt}\n  {res_ckpt}")

    device = _device()
    meta = np.load(cache_dir / "meta.npz", allow_pickle=True)
    n = len(meta["sample_idx"])
    splits = make_splits(n, seed=args.seed)
    # Fit norms on train (must match training); evaluate on test.
    train_ds = ResidualDeepONetDataset(
        cache_dir, splits.train, target=target, trunk_set=trunk_set  # type: ignore[arg-type]
    )
    test_ds = ResidualDeepONetDataset(
        cache_dir, splits.test, target=target, trunk_set=trunk_set  # type: ignore[arg-type]
    )
    stats = fit_and_apply_norms(train_ds, test_ds)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    n_rec, n_freq = test_ds.n_rec, len(test_ds.f_idx)
    freq = test_ds.freq_s

    packs: Dict[str, Dict[str, np.ndarray]] = {}
    aggregates: Dict[str, Dict[str, float]] = {}
    per_sample: Dict[str, Dict[str, np.ndarray]] = {}

    for label, path, enc in [
        ("Conv", conv_ckpt, "conv"),
        ("ResUNet", res_ckpt, "resunet"),
    ]:
        model, _ = _load_model(path, field_encoder=enc)  # type: ignore[arg-type]
        model = model.to(device)
        pack = collect_predictions(
            model,
            test_loader,
            stats,
            device,
            n_rec=n_rec,
            n_freq=n_freq,
        )
        packs[label] = pack
        aggregates[label] = aggregate(pack)
        per_sample[label] = per_sample_metrics(pack)
        print(f"{label}: {aggregates[label]}", flush=True)

    out_json = config.RESULTS_DIR / f"encoder_compare_{target}_{args.cache_tag}.json"
    out_json.write_text(
        json.dumps(
            {
                "target": target,
                "cache_tag": args.cache_tag,
                "aggregate": aggregates,
                "per_sample_summary": {
                    lab: {
                        k: {
                            "mean": float(np.mean(v)),
                            "median": float(np.median(v)),
                            "std": float(np.std(v)),
                        }
                        for k, v in mets.items()
                    }
                    for lab, mets in per_sample.items()
                },
            },
            indent=2,
        )
    )
    print(f"Wrote {out_json}", flush=True)

    plot_metric_boxes(
        per_sample, config.RESULTS_DIR / f"encoder_box_{target}_{args.cache_tag}.png"
    )
    plot_metric_hists(
        per_sample, config.RESULTS_DIR / f"encoder_hist_{target}_{args.cache_tag}.png"
    )
    plot_tf_curves(
        packs,
        freq,
        config.RESULTS_DIR / f"encoder_tf_{target}_{args.cache_tag}.png",
    )
    plot_residual_curves(
        packs,
        freq,
        config.RESULTS_DIR / f"encoder_residual_{target}_{args.cache_tag}.png",
    )
    print("Plots written to", config.RESULTS_DIR, flush=True)


if __name__ == "__main__":
    main()
