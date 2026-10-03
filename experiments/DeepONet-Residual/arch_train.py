#!/usr/bin/env python3
"""Laptop-first residual bake-off: n-ladder, train recipe, FNO-on-R, recorder GNO."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

import config
from data import CombinedResidualDataset, ResidualDeepONetDataset, stoch_dim
from mix_ladder import is_iid_only_mix, mix_test_parts, mix_train_parts, mix_val_parts
from model import build_model
from train import train_from_datasets

ARCH_DIR = config.RESULTS_DIR / "arch_train"


def _ds(
    cache: Path,
    idx: np.ndarray,
    *,
    n_freq: int,
    serial: bool,
    freq_sample: str = "log",
    trunk_scales: int = 1,
    stoch_layout: str = "xi_cov",
    fstar_kind: str = "legacy",
    x_coord: str = "x_over_lambda",
    nom_variant: str = "default",
    query_split: str = "all",
    support_stride: int = 0,
) -> ResidualDeepONetDataset:
    return ResidualDeepONetDataset(
        cache,
        idx,
        target="R_nom",
        trunk_set="full",
        n_freq=n_freq,
        serial_tf1d=serial,
        freq_sample=freq_sample,
        trunk_scales=trunk_scales,
        stoch_layout=stoch_layout,
        fstar_kind=fstar_kind,
        x_coord=x_coord,
        nom_variant=nom_variant,
        query_split=query_split,
        support_stride=support_stride,
    )


def _combined(
    parts: list[tuple[str, Path, np.ndarray]],
    *,
    n_freq: int,
    serial: bool,
    freq_sample: str = "log",
    trunk_scales: int = 1,
    stoch_layout: str = "xi_cov",
    fstar_kind: str = "legacy",
    x_coord: str = "x_over_lambda",
    nom_variant: str = "default",
    query_split: str = "all",
    support_stride: int = 0,
) -> CombinedResidualDataset:
    return CombinedResidualDataset(
        [
            _ds(
                c,
                i,
                n_freq=n_freq,
                serial=serial,
                freq_sample=freq_sample,
                trunk_scales=trunk_scales,
                stoch_layout=stoch_layout,
                fstar_kind=fstar_kind,
                x_coord=x_coord,
                nom_variant=nom_variant,
                query_split=query_split,
                support_stride=support_stride,
            )
            for _, c, i in parts
        ],
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
    physics_tokens: bool = False,
    learned_1d: bool = False,
    gated: bool = False,
    gate_sparsity: float = 0.0,
    log_residual: bool = False,
    radial_loss_weight: float = 0.0,
    pod_readout: bool = False,
    boost_ckpt: Path | None = None,
    boost_fno_kind: str = "loglo",
    boost_width: int | None = None,
    init_ckpt: Path | None = None,
    freeze_gno: bool = False,
    freeze_fno: bool = False,
    col_enc_depth_tokens: int = 1,
    val_monitor: str = "smooth_l1",
    trunk_scales: int = 1,
    band_weight: float = 0.0,
    aux_logspec_full: float = 0.0,
    seed: int = config.SEED,
    freq_sample: str = "log",
    aux_logspec: float = 0.0,
    aux_dct_below: float = 0.0,
    dct_cutoff: int = 16,
    smooth_l1_beta: float | None = None,
    n_mscale_trunk: int = 1,
    n_mscale_branch: int = 1,
    col_enc: str = "conv",
    stoch_inject: str = "mlp",
    fuse_kind: str = "mlp",
    stoch_layout: str = "xi_cov",
    fstar_kind: str = "legacy",
    x_coord: str = "x_over_lambda",
    nom_variant: str = "default",
    gno_rh_dilate: bool = False,
    encoder_lr: float | None = None,
    query_split: str = "all",
    support_stride: int = 0,
    kernel_k: int = 2,
    latent_fno: bool = False,
    n_latent: int = 32,
) -> dict[str, Any]:
    from residual_signed import build_signed_cache

    if mix_tag in ("M2100",):
        n3000 = config.CACHE_DIR / "n3000_seed42"
        if not (n3000 / "r_nom_signed.npy").is_file():
            print("[arch] building n3000 signed cache (Haskell once)", flush=True)
            build_signed_cache("n3000_seed42")
    if mix_tag in ("IID2000", "M1400"):
        n2000 = config.CACHE_DIR / "n2000_seed42"
        if not (n2000 / "r_nom_signed.npy").is_file():
            n7680 = config.CACHE_DIR / "n7680_seed42"
            if (n7680 / "r_nom_signed.npy").is_file():
                from ood_signed_cache import materialize_signed_from_parent

                print("[arch] materializing n2000 from n7680 (no Haskell)", flush=True)
                materialize_signed_from_parent("n2000_seed42", "n7680_seed42")
            else:
                print("[arch] building n2000 signed cache (Haskell once)", flush=True)
                build_signed_cache("n2000_seed42")
    if mix_tag in ("M7680", "IID7680"):
        n7680 = config.CACHE_DIR / "n7680_seed42"
        if not (n7680 / "r_nom_signed.npy").is_file():
            print("[arch] building n7680 signed cache (Haskell once)", flush=True)
            build_signed_cache("n7680_seed42")
    # Nested tests stay seed-42 even when --seed wobbles training RNG.
    iid_only = is_iid_only_mix(mix_tag)
    train_parts = mix_train_parts(mix_tag)
    val_parts = mix_val_parts(iid_only=iid_only)
    tests = mix_test_parts()
    print(
        f"[arch] {run_name} mix={mix_tag} n_train_parts="
        f"{[(n, len(i)) for n, _, i in train_parts]}",
        flush=True,
    )
    train_ds = _combined(
        train_parts,
        n_freq=n_freq_train,
        serial=serial,
        freq_sample=freq_sample,
        trunk_scales=trunk_scales,
        stoch_layout=stoch_layout,
        fstar_kind=fstar_kind,
        x_coord=x_coord,
        nom_variant=nom_variant,
        query_split=query_split,
        support_stride=support_stride,
    )
    val_ds = _combined(
        val_parts,
        n_freq=n_freq_train,
        serial=serial,
        freq_sample=freq_sample,
        trunk_scales=trunk_scales,
        stoch_layout=stoch_layout,
        fstar_kind=fstar_kind,
        x_coord=x_coord,
        nom_variant=nom_variant,
        query_split=query_split,
        support_stride=support_stride,
    )
    extra = {
        dname: _ds(
            c,
            i,
            n_freq=n_freq_eval,
            serial=serial,
            freq_sample="log",
            trunk_scales=trunk_scales,
            stoch_layout=stoch_layout,
            fstar_kind=fstar_kind,
            x_coord=x_coord,
            nom_variant=nom_variant,
            query_split="all",
            support_stride=support_stride,
        )
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
        seed=seed,
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
        aux_logspec=aux_logspec,
        aux_dct_below=aux_dct_below,
        dct_cutoff=dct_cutoff,
        smooth_l1_beta=smooth_l1_beta,
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
        physics_tokens=physics_tokens,
        geom_flag_dim=config.GEOM_FLAG_DIM if physics_tokens else 0,
        learned_1d=learned_1d,
        gated=gated,
        gate_sparsity=gate_sparsity,
        leftover_tau=config.LEFTOVER_GATE_TAU,
        learned_1d_weight=config.LEARNED_1D_WEIGHT if learned_1d else 0.0,
        log_residual=log_residual,
        radial_loss_weight=radial_loss_weight,
        pod_readout=pod_readout,
        mix_train_parts=train_parts,
        boost_ckpt=boost_ckpt,
        boost_fno_kind=boost_fno_kind,
        boost_width=boost_width,
        init_ckpt=init_ckpt,
        freeze_gno=freeze_gno,
        freeze_fno=freeze_fno,
        col_enc_depth_tokens=col_enc_depth_tokens,
        val_monitor=val_monitor,
        freq_sample=freq_sample,
        band_weight=band_weight,
        aux_logspec_full=aux_logspec_full,
        n_mscale_trunk=n_mscale_trunk,
        n_mscale_branch=n_mscale_branch,
        col_enc=col_enc,
        stoch_inject=stoch_inject,
        fuse_kind=fuse_kind,
        gno_rh_dilate=gno_rh_dilate,
        encoder_lr=encoder_lr,
        kernel_k=kernel_k,
        latent_fno=latent_fno,
        n_latent=n_latent,
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
        "--mix",
        choices=["M700", "M1400", "M2100", "M7680", "IID2000", "IID7680"],
        default="M1400",
    )
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument(
        "--encoder",
        choices=["conv", "resunet", "gno", "attn", "gat", "identity", "kernel"],
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
        choices=[
            "vanilla",
            "ufno",
            "ffno",
            "afno",
            "wno",
            "fno1d",
            "loglo",
            "tf",
            "band2",
        ],
        default="vanilla",
        help="vanilla FNO, U-FNO, F-FNO, leftover DualPathLOGLO, axial leftover TF, or band2 Hz split.",
    )
    p.add_argument("--n-freq-train", type=int, default=config.N_FREQ_TRAIN)
    p.add_argument("--n-freq-eval", type=int, default=config.N_FREQ_EVAL)
    p.add_argument("--iid-frac", type=float, default=None)
    p.add_argument(
        "--init-ckpt",
        type=Path,
        default=None,
        help="Load leftover weights before training (fine-tune). Reuses checkpoint z-score stats.",
    )
    p.add_argument(
        "--freeze-gno",
        action="store_true",
        help="Freeze col_enc + chain GNO; train fuse/trunk/leftover head.",
    )
    p.add_argument(
        "--freeze-fno",
        action="store_true",
        help="Freeze leftover FNO head; train col_enc/GNO/fuse/trunk.",
    )
    p.add_argument(
        "--gno-rh-dilate",
        action="store_true",
        help="Add rH-scaled dilation skip on the recorder GNO (unfreeze GNO; lower encoder LR).",
    )
    p.add_argument(
        "--encoder-lr",
        type=float,
        default=None,
        help="AdamW LR for col_enc/gno when unfrozen. Default 0.1×--lr if --gno-rh-dilate.",
    )
    p.add_argument(
        "--col-enc-depth-tokens",
        type=int,
        default=config.COL_ENC_DEPTH_TOKENS,
        help="Depth bins kept by the column encoder (1 = ship global pool).",
    )
    p.add_argument(
        "--val-monitor",
        choices=["smooth_l1", "three_layer"],
        default="smooth_l1",
        help="Early-stop and LR plateau metric. three_layer = val ood_three_layer rel_l2_TF.",
    )
    p.add_argument(
        "--trunk-scales",
        type=int,
        default=1,
        help="Octave-spaced Fourier harmonics in the trunk (1 = shipped single pair).",
    )
    p.add_argument(
        "--mscale-trunk",
        type=int,
        default=1,
        dest="n_mscale_trunk",
        help="Parallel scaled trunk subnets (1 = shipped single TrunkMLP).",
    )
    p.add_argument(
        "--mscale-branch",
        type=int,
        default=1,
        dest="n_mscale_branch",
        help="Shared-encoder branch scales (1 = shipped single fuse).",
    )
    p.add_argument(
        "--stoch-inject",
        choices=["mlp", "concat"],
        default="mlp",
        help="GINO: Linear(stoch→latent)+GELU, or concat raw stoch into the fuse.",
    )
    p.add_argument(
        "--fuse",
        choices=["mlp", "add"],
        default="mlp",
        dest="fuse_kind",
        help="GINO: fuse MLP on cat(nodes,s), or add (requires --stoch-inject mlp).",
    )
    p.add_argument(
        "--stoch-layout",
        choices=["xi_cov", "cov_only", "legacy20", "xi_field_acf"],
        default="xi_cov",
        help="Stochastic branch: ξ+CoV (shipped), CoV only, legacy 20-d, or field-FFT ξ+CoV+ACF.",
    )
    p.add_argument(
        "--fstar",
        choices=["legacy", "tts"],
        default="legacy",
        dest="fstar_kind",
        help="legacy = f H / vs_col; tts = f / f0 with f0 = 1/(4 Σ H_i/Vs_i).",
    )
    p.add_argument(
        "--x-coord",
        choices=["x_over_lambda", "xH"],
        default="x_over_lambda",
        help="Lateral trunk coordinate. xH replaces x/λ (redundant with f* on flat sites).",
    )
    p.add_argument(
        "--nom-variant",
        choices=["default", "sample_xi"],
        default="default",
        help="default = cached TF_1D_nom (ξ=0.05); sample_xi = tf1d_nom_xi extras.",
    )
    p.add_argument(
        "--query-split",
        choices=["all", "even", "interior"],
        default="all",
        help="Train/val labeled stations. even = hold out odds; interior drops r0/r20.",
    )
    p.add_argument(
        "--support-stride",
        type=int,
        default=None,
        help="Kernel support column stride on the 500 m strip (default 5 if --encoder kernel).",
    )
    p.add_argument(
        "--kernel-k",
        type=int,
        default=2,
        help="Nearest support columns for kernel GNO (physical |Δx|).",
    )
    p.add_argument(
        "--latent-fno",
        action="store_true",
        help="Phase 1b: FNO on a fixed latent x-grid, then decode to query x.",
    )
    p.add_argument(
        "--n-latent",
        type=int,
        default=32,
        help="Latent x-grid size for --latent-fno.",
    )
    p.add_argument(
        "--col-enc",
        choices=["conv", "mlp", "attn"],
        default="conv",
        help="GINO depth encoder. mlp/attn pool to 16 bins (not --encoder attn).",
    )
    p.add_argument(
        "--band-weight",
        type=float,
        default=0.0,
        help="Weight on the band-normalized residual MSE (equal low/mid/high weight).",
    )
    p.add_argument(
        "--aux-logspec-full",
        type=float,
        default=0.0,
        help="Weight on trough-retaining log-amplitude MSE (matches the misfit score).",
    )
    p.add_argument("--aux-tf-rel-l2", type=float, default=0.0)
    p.add_argument("--aux-peak-band", type=float, default=0.0)
    p.add_argument(
        "--aux-logspec",
        type=float,
        default=0.0,
        help="Weight on trough-windowed log|TF| rel L2 (Wave A arm C).",
    )
    p.add_argument(
        "--aux-dct-below",
        type=float,
        default=0.0,
        help="Weight on DCT |R| L1 below --dct-cutoff (Wave A arm E; freq axis).",
    )
    p.add_argument(
        "--dct-cutoff",
        type=int,
        default=16,
        help="FNO freq-mode cutoff for --aux-dct-below (n_modes[1]).",
    )
    p.add_argument(
        "--smooth-l1-beta",
        type=float,
        default=None,
        help="SmoothL1 beta override (default config 1.0; arm B uses 0.1).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=config.SEED,
        help="Training RNG / iid-frac sampler wobble. Mix train/val/test slices stay seed-42.",
    )
    p.add_argument(
        "--freq-sample",
        choices=["log", "band"],
        default="log",
        help="Train/val frequency queries: log-spaced or equal counts per Hz band.",
    )
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
    p.add_argument(
        "--gated",
        action="store_true",
        help="Per-recorder fail-soft gate: TF = TF1D + sigma(g)*R.",
    )
    p.add_argument(
        "--physics-tokens",
        action="store_true",
        help="Fuse Haskell log|TF1D| + dip/layer flags into the GNO encoder.",
    )
    p.add_argument(
        "--learned-1d",
        action="store_true",
        help="Aux head matching Haskell TF1D (kill if Pearson vs Haskell < 0.98).",
    )
    p.add_argument(
        "--gate-sparsity",
        type=float,
        default=config.GATE_SPARSITY,
        help="L1 on gate for small-leftover files (used with --gated).",
    )
    p.add_argument(
        "--log-residual",
        action="store_true",
        help="Predict r such that TF = TF1D * exp(r) (log-multiplicative leftover).",
    )
    p.add_argument(
        "--radial-loss",
        type=float,
        default=None,
        help="Weight on LOGLO radial binned spectral loss (default 0.25 with --log-residual).",
    )
    p.add_argument(
        "--pod-readout",
        action="store_true",
        help="POD-DeepONet readout of mix-train leftover R (not TF).",
    )
    p.add_argument(
        "--boost-ckpt",
        type=Path,
        default=None,
        help="Freeze this leftover checkpoint and train a tiny booster (E3b).",
    )
    p.add_argument(
        "--boost-fno-kind",
        choices=["vanilla", "ufno", "ffno", "afno", "wno", "fno1d", "loglo"],
        default="loglo",
    )
    p.add_argument(
        "--boost-width",
        type=int,
        default=None,
        help="Booster FNO width (default: fno_width // 2).",
    )
    p.add_argument("--dump-m700", action="store_true")
    p.add_argument(
        "--probe-vram",
        action="store_true",
        help="Forward/backward VRAM probe, then exit (no training).",
    )
    args = p.parse_args()
    if args.fuse_kind == "add" and args.stoch_inject != "mlp":
        raise SystemExit("--fuse add requires --stoch-inject mlp")
    if args.fno_kind == "loglo" and "--fno-layers" not in sys.argv:
        args.fno_layers = 1
    if is_iid_only_mix(args.mix) and args.val_monitor == "three_layer":
        raise SystemExit(
            "IID* mixes have no three-layer val slice; use --val-monitor smooth_l1"
        )
    config.set_local_ood_env()
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
            elif args.encoder == "kernel":
                run_name = f"{args.mix}_kernel"
            if args.fno:
                run_name = f"{run_name}_fno"
                if args.fno_kind != "vanilla":
                    run_name = f"{run_name}_{args.fno_kind}"
            if args.latent_fno:
                run_name = f"{run_name}_latfno"
        if args.iid_frac is not None:
            run_name = f"{run_name}_iid{int(100 * args.iid_frac)}"
        if args.freeze_gno:
            run_name = f"{run_name}_ftgno"
        if args.freeze_fno:
            run_name = f"{run_name}_ftfno"
        if int(args.col_enc_depth_tokens) != int(config.COL_ENC_DEPTH_TOKENS):
            run_name = f"{run_name}_z{int(args.col_enc_depth_tokens)}"
        if args.val_monitor == "three_layer":
            run_name = f"{run_name}_3lstop"
        if args.aux_peak_band:
            run_name = f"{run_name}_peak"
        if args.aux_logspec:
            run_name = f"{run_name}_logspec"
        if args.aux_dct_below:
            run_name = f"{run_name}_dct{int(args.dct_cutoff)}"
        if args.smooth_l1_beta is not None:
            run_name = f"{run_name}_b{args.smooth_l1_beta:g}"
        if args.freq_sample == "band":
            run_name = f"{run_name}_bandf"
        if args.seed != config.SEED:
            run_name = f"{run_name}_s{args.seed}"
        if args.gated:
            run_name = f"{run_name}_gated"
        if args.physics_tokens:
            run_name = f"{run_name}_phys"
        if args.learned_1d:
            run_name = f"{run_name}_1d"
        if args.log_residual:
            run_name = f"{run_name}_logR"
        if args.pod_readout:
            run_name = f"{run_name}_pod"
        if args.boost_ckpt is not None:
            run_name = f"{run_name}_boost"
        if int(args.n_mscale_trunk) > 1:
            run_name = f"{run_name}_mT{int(args.n_mscale_trunk)}"
        if int(args.n_mscale_branch) > 1:
            run_name = f"{run_name}_mB{int(args.n_mscale_branch)}"
        if args.stoch_inject != "mlp":
            run_name = f"{run_name}_{args.stoch_inject}stoch"
        if args.fuse_kind != "mlp":
            run_name = f"{run_name}_{args.fuse_kind}fuse"
        if args.col_enc != "conv":
            run_name = f"{run_name}_col{args.col_enc}"
        if int(args.trunk_scales) > 1:
            run_name = f"{run_name}_ff{int(args.trunk_scales)}"
        if args.stoch_layout != "xi_cov":
            run_name = f"{run_name}_{args.stoch_layout}"
        if args.gno_rh_dilate:
            run_name = f"{run_name}_rhdilate"
        if args.fstar_kind != "legacy":
            run_name = f"{run_name}_f0{args.fstar_kind}"
        if args.x_coord != "x_over_lambda":
            run_name = f"{run_name}_{args.x_coord}"
        if args.nom_variant != "default":
            run_name = f"{run_name}_{args.nom_variant}"
        if args.query_split != "all":
            run_name = f"{run_name}_q{args.query_split}"
        if args.latent_fno and args.run_name is None and "_latfno" not in run_name:
            run_name = f"{run_name}_latfno"
    radial_w = args.radial_loss
    if radial_w is None:
        radial_w = config.RADIAL_LOSS_WEIGHT if args.log_residual else 0.0
    booster_kind = args.boost_fno_kind
    train_kind = args.fno_kind
    if args.boost_ckpt is not None:
        train_kind = booster_kind
    residual_fno = bool(args.fno)
    if args.encoder == "kernel" and residual_fno and not args.latent_fno:
        print(
            "[arch] FNO-on-R is off for kernel GNO (variable queries). "
            "Use --latent-fno for Phase 1b.",
            flush=True,
        )
        residual_fno = False
    support_stride = args.support_stride
    if support_stride is None:
        support_stride = config.SUPPORT_STRIDE if args.encoder == "kernel" else 0
    run_mix(
        mix_tag=args.mix,
        run_name=run_name,
        encoder=args.encoder,
        serial=args.serial,
        residual_fno=residual_fno,
        iid_frac=args.iid_frac,
        aux_tf_rel_l2=args.aux_tf_rel_l2,
        aux_peak_band=args.aux_peak_band,
        aux_logspec=args.aux_logspec,
        aux_dct_below=args.aux_dct_below,
        dct_cutoff=args.dct_cutoff,
        smooth_l1_beta=args.smooth_l1_beta,
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
        fno_kind=train_kind,
        lr_sched_factor=args.lr_sched_factor,
        lr_sched_patience=args.lr_sched_patience,
        lr_sched_min=args.lr_sched_min,
        use_lr_sched=args.lr_sched,
        physics_tokens=args.physics_tokens,
        learned_1d=args.learned_1d,
        gated=args.gated,
        gate_sparsity=args.gate_sparsity if args.gated else 0.0,
        log_residual=args.log_residual,
        radial_loss_weight=float(radial_w),
        pod_readout=args.pod_readout,
        boost_ckpt=args.boost_ckpt,
        boost_fno_kind=booster_kind,
        boost_width=args.boost_width,
        init_ckpt=args.init_ckpt,
        freeze_gno=args.freeze_gno,
        freeze_fno=args.freeze_fno,
        col_enc_depth_tokens=args.col_enc_depth_tokens,
        val_monitor=args.val_monitor,
        seed=args.seed,
        freq_sample=args.freq_sample,
        trunk_scales=args.trunk_scales,
        band_weight=args.band_weight,
        aux_logspec_full=args.aux_logspec_full,
        n_mscale_trunk=args.n_mscale_trunk,
        n_mscale_branch=args.n_mscale_branch,
        col_enc=args.col_enc,
        stoch_inject=args.stoch_inject,
        fuse_kind=args.fuse_kind,
        stoch_layout=args.stoch_layout,
        fstar_kind=args.fstar_kind,
        x_coord=args.x_coord,
        nom_variant=args.nom_variant,
        gno_rh_dilate=args.gno_rh_dilate,
        encoder_lr=args.encoder_lr,
        query_split=args.query_split,
        support_stride=int(support_stride),
        kernel_k=args.kernel_k,
        latent_fno=args.latent_fno,
        n_latent=args.n_latent,
    )


if __name__ == "__main__":
    main()
