#!/usr/bin/env python3
"""Laptop-first residual bake-off: n-ladder, train recipe, FNO-on-R, recorder GNO."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

import config
from data import CombinedResidualDataset, ResidualDeepONetDataset, stoch_dim
from mix_ladder import mix_test_parts, mix_train_parts, mix_val_parts
from model import build_model
from train import train_from_datasets

ARCH_DIR = config.RESULTS_DIR / "arch_train"


def _set_local_ood_env() -> None:
    local = Path(__file__).resolve().parents[2] / "data" / "gifno_screen"
    if (local / "ood_dipping").is_dir():
        os.environ.setdefault("GIFNO_OOD_DIPPING", str(local / "ood_dipping"))
    if (local / "ood_three_layer").is_dir():
        os.environ.setdefault("GIFNO_OOD_THREE_LAYER", str(local / "ood_three_layer"))


def _ds(
    cache: Path, idx: np.ndarray, *, n_freq: int, serial: bool
) -> ResidualDeepONetDataset:
    return ResidualDeepONetDataset(
        cache,
        idx,
        target="R_nom",
        trunk_set="full",
        n_freq=n_freq,
        serial_tf1d=serial,
    )


def _combined(
    parts: list[tuple[str, Path, np.ndarray]],
    *,
    n_freq: int,
    serial: bool,
) -> CombinedResidualDataset:
    return CombinedResidualDataset(
        [_ds(c, i, n_freq=n_freq, serial=serial) for _, c, i in parts],
        domain_names=[name for name, _, _ in parts],
    )


def _parse_modes(text: str) -> tuple[int, int]:
    parts = [int(x.strip()) for x in str(text).split(",")]
    if len(parts) != 2:
        raise ValueError(f"expected two FNO modes like 8,32; got {text!r}")
    return parts[0], parts[1]


def run_mix(
    *,
    mix_tag: str,
    run_name: str,
    encoder: str,
    serial: bool,
    residual_fno: bool,
    iid_frac: float | None,
    aux_tf_rel_l2: float,
    aux_peak_band: float,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    use_wandb: bool,
    n_freq_train: int = config.N_FREQ_TRAIN,
    n_freq_eval: int = config.N_FREQ_EVAL,
    fno_width: int = config.FNO_WIDTH,
    fno_n_modes: tuple[int, int] = config.FNO_N_MODES,
    fno_n_layers: int = config.FNO_N_LAYERS,
    fno_kind: str = "vanilla",
    lr_sched_factor: float = config.LR_SCHED_FACTOR,
    lr_sched_patience: int = config.LR_SCHED_PATIENCE,
    lr_sched_min: float = config.LR_SCHED_MIN,
    use_lr_sched: bool = True,
) -> dict[str, Any]:
    from residual_signed import build_signed_cache

    if mix_tag == "M2100":
        n3000 = config.CACHE_DIR / "n3000_seed42"
        if not (n3000 / "r_nom_signed.npy").is_file():
            print("[arch] building n3000 signed cache (Haskell once)", flush=True)
            build_signed_cache("n3000_seed42")
    if mix_tag == "M7680":
        n7680 = config.CACHE_DIR / "n7680_seed42"
        if not (n7680 / "r_nom_signed.npy").is_file():
            print("[arch] building n7680 signed cache (Haskell once)", flush=True)
            build_signed_cache("n7680_seed42")
    train_parts = mix_train_parts(mix_tag)
    val_parts = mix_val_parts()
    tests = mix_test_parts()
    print(
        f"[arch] {run_name} mix={mix_tag} n_train_parts="
        f"{[(n, len(i)) for n, _, i in train_parts]}",
        flush=True,
    )
    train_ds = _combined(train_parts, n_freq=n_freq_train, serial=serial)
    val_ds = _combined(val_parts, n_freq=n_freq_train, serial=serial)
    extra = {
        dname: _ds(c, i, n_freq=n_freq_eval, serial=serial)
        for dname, (c, i) in tests.items()
    }
    return train_from_datasets(
        train_ds=train_ds,
        val_ds=val_ds,
        extra_tests=extra,
        target="R_nom",
        branch_mode="single",
        trunk_set="full",
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=config.SEED,
        run_name=run_name,
        patience=patience,
        field_encoder=encoder,  # type: ignore[arg-type]
        n_freq_train=n_freq_train,
        n_freq_eval=n_freq_eval,
        serial_tf1d=serial,
        use_wandb=use_wandb,
        iid_frac=iid_frac,
        aux_tf_rel_l2=aux_tf_rel_l2,
        aux_peak_band=aux_peak_band,
        residual_fno=residual_fno,
        fno_width=fno_width,
        fno_n_modes=fno_n_modes,
        fno_n_layers=fno_n_layers,
        fno_kind=fno_kind,
        mix_tag=mix_tag,
        lr_sched_factor=lr_sched_factor,
        lr_sched_patience=lr_sched_patience,
        lr_sched_min=lr_sched_min,
        use_lr_sched=use_lr_sched,
    )


def probe_vram(
    batch_size: int,
    *,
    residual_fno: bool,
    encoder: str,
    serial: bool,
    n_freq: int = config.N_FREQ_TRAIN,
    fno_width: int = config.FNO_WIDTH,
    fno_n_modes: tuple[int, int] = config.FNO_N_MODES,
    fno_n_layers: int = config.FNO_N_LAYERS,
    fno_kind: str = "vanilla",
) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trunk_dim = 5 if serial else 4
    n_rec = config.N_LATERAL
    model = build_model(
        "single",
        field_channels=config.FIELD_CHANNELS,
        stoch_dim=stoch_dim(),
        trunk_dim=trunk_dim,
        latent_dim=config.LATENT_DIM,
        field_hidden=config.FIELD_HIDDEN,
        branch_hidden=config.BRANCH_HIDDEN,
        trunk_hidden=config.TRUNK_HIDDEN,
        trunk_layers=config.TRUNK_LAYERS,
        field_encoder=encoder,  # type: ignore[arg-type]
        residual_fno=residual_fno,
        n_rec=n_rec,
        fno_width=fno_width,
        fno_n_modes=fno_n_modes,
        fno_n_layers=fno_n_layers,
        fno_kind=fno_kind,  # type: ignore[arg-type]
    ).to(device)
    fields = torch.randn(batch_size, 3, config.NZ_MAX, n_rec, device=device)
    stoch = torch.randn(batch_size, stoch_dim(), device=device)
    trunk = torch.randn(batch_size, n_rec * n_freq, trunk_dim, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    for _ in range(5):
        pred = model(fields, stoch, trunk)
        loss = pred.square().mean()
        loss.backward()
        model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
    else:
        alloc = reserved = peak = 0.0
    out = {
        "batch_size": float(batch_size),
        "device": str(device),
        "alloc_gb": alloc,
        "reserved_gb": reserved,
        "peak_gb": peak,
        "n_params": float(sum(p.numel() for p in model.parameters())),
    }
    print(json.dumps(out, indent=2), flush=True)
    return out


def dump_m700_baseline() -> dict[str, Any]:
    """Reuse shipped serial P3 mix as M700 control (same recipe, already trained)."""
    ckpt = config.DEFAULT_CHECKPOINT
    blob: dict[str, Any] = {}
    if ckpt.is_file():
        packed = torch.load(ckpt, map_location="cpu", weights_only=False)
        blob = {
            "name": "M700_serial",
            "checkpoint": str(ckpt),
            "test_by_domain": packed.get("test_by_domain"),
            "n_train": 2044,
            "reused": True,
        }
    json_path = config.RESULTS_DIR / "domain_study" / "architectures.json"
    if json_path.is_file() and not blob.get("test_by_domain"):
        arch = json.loads(json_path.read_text())
        serial = arch.get("serial") or {}
        blob = {
            "name": "M700_serial",
            "checkpoint": serial.get("checkpoint", str(ckpt)),
            "test_by_domain": serial.get("test_by_domain"),
            "n_train": 2044,
            "reused": True,
        }
    ARCH_DIR.mkdir(parents=True, exist_ok=True)
    (ARCH_DIR / "M700_serial.json").write_text(json.dumps(blob, indent=2, default=str))
    print(f"[arch] wrote M700 baseline → {ARCH_DIR / 'M700_serial.json'}", flush=True)
    return blob


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mix", choices=["M700", "M1400", "M2100", "M7680"], default="M1400"
    )
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument(
        "--encoder",
        choices=["conv", "resunet", "gno", "attn", "gat"],
        default="resunet",
    )
    p.add_argument("--serial", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fno", action="store_true")
    p.add_argument("--fno-width", type=int, default=config.FNO_WIDTH)
    p.add_argument(
        "--fno-modes", type=str, default=",".join(str(x) for x in config.FNO_N_MODES)
    )
    p.add_argument("--fno-layers", type=int, default=config.FNO_N_LAYERS)
    p.add_argument(
        "--fno-kind",
        choices=["vanilla", "ufno", "ffno", "afno", "wno", "fno1d"],
        default="vanilla",
        help="vanilla FNO, U-FNO (local conv), or factorized F-FNO.",
    )
    p.add_argument("--n-freq-train", type=int, default=config.N_FREQ_TRAIN)
    p.add_argument("--n-freq-eval", type=int, default=config.N_FREQ_EVAL)
    p.add_argument("--iid-frac", type=float, default=None)
    p.add_argument("--aux-tf-rel-l2", type=float, default=0.0)
    p.add_argument("--aux-peak-band", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--patience", type=int, default=config.PATIENCE)
    p.add_argument(
        "--lr-sched",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ReduceLROnPlateau on val SmoothL1; restore best-val weights at the end.",
    )
    p.add_argument("--lr-sched-factor", type=float, default=config.LR_SCHED_FACTOR)
    p.add_argument("--lr-sched-patience", type=int, default=config.LR_SCHED_PATIENCE)
    p.add_argument("--lr-sched-min", type=float, default=config.LR_SCHED_MIN)
    p.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=config.WANDB_DEFAULT,
    )
    p.add_argument("--probe-vram", action="store_true")
    p.add_argument("--dump-m700", action="store_true")
    args = p.parse_args()
    _set_local_ood_env()
    ARCH_DIR.mkdir(parents=True, exist_ok=True)
    fno_modes = _parse_modes(args.fno_modes)
    if args.dump_m700:
        dump_m700_baseline()
        return
    if args.probe_vram:
        probe_vram(
            args.batch_size,
            residual_fno=args.fno,
            encoder=args.encoder,
            serial=args.serial,
            n_freq=args.n_freq_train,
            fno_width=args.fno_width,
            fno_n_modes=fno_modes,
            fno_n_layers=args.fno_layers,
            fno_kind=args.fno_kind,
        )
        return
    run_name = args.run_name or f"{args.mix}_{args.encoder}"
    if args.run_name is None:
        run_name = f"{args.mix}_{args.encoder}"
        if args.serial:
            run_name = f"{args.mix}_serial"
            if args.encoder == "gno":
                run_name = f"{args.mix}_gno"
            elif args.encoder == "attn":
                run_name = f"{args.mix}_attn"
            elif args.encoder == "gat":
                run_name = f"{args.mix}_gat"
            if args.fno:
                run_name = f"{run_name}_fno"
                if args.fno_kind != "vanilla":
                    run_name = f"{run_name}_{args.fno_kind}"
        if args.iid_frac is not None:
            run_name = f"{run_name}_iid{int(100 * args.iid_frac)}"
        if args.aux_tf_rel_l2:
            run_name = f"{run_name}_auxL2"
        if args.aux_peak_band:
            run_name = f"{run_name}_peak"
    run_mix(
        mix_tag=args.mix,
        run_name=run_name,
        encoder=args.encoder,
        serial=args.serial,
        residual_fno=args.fno,
        iid_frac=args.iid_frac,
        aux_tf_rel_l2=args.aux_tf_rel_l2,
        aux_peak_band=args.aux_peak_band,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        use_wandb=args.wandb,
        n_freq_train=args.n_freq_train,
        n_freq_eval=args.n_freq_eval,
        fno_width=args.fno_width,
        fno_n_modes=fno_modes,
        fno_n_layers=args.fno_layers,
        fno_kind=args.fno_kind,
        lr_sched_factor=args.lr_sched_factor,
        lr_sched_patience=args.lr_sched_patience,
        lr_sched_min=args.lr_sched_min,
        use_lr_sched=args.lr_sched,
    )


if __name__ == "__main__":
    main()
