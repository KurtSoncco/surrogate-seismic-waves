"""Train / evaluate signed-residual DeepONet."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import config
import numpy as np
import torch
from model import (
    BranchMode,
    FieldEncoderKind,
    FrozenBoost,
    apply_query_freq,
    build_model,
    freeze_fno_head,
    freeze_gno_encoder,
    gno_core,
)
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm, trange
from wandb_util import finish_wandb, init_wandb, log_wandb, summary_wandb
from unified_metrics import (
    flat_pearson,
    flat_r2,
    flat_rel_l2,
    pearson_across_freq,
    score_leftover_batch,
    tf_from_residual,
)
from spectral_aux import (
    DCT_CUTOFF,
    band_normalized_loss,
    dct_below_cutoff_loss,
    logspec_full_loss,
    trough_safe_logspec_loss,
)

from data import (
    ResidualDeepONetDataset,
    TargetName,
    TrunkSet,
    iid_resample_sampler,
    make_splits,
    remap_stoch_last_dim,
    stoch_dim_from_dataset,
    stoch_layout_from_dim,
)


def _build_signed_cache(*args, **kwargs):
    from residual_signed import build_signed_cache

    return build_signed_cache(*args, **kwargs)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _stack_key(ds, key: str) -> torch.Tensor:
    return torch.stack([ds._cache[i][key] for i in range(len(ds))], dim=0)


def fit_norms(train_ds) -> dict[str, torch.Tensor]:
    """Z-score stats from the training split only."""
    stoch = _stack_key(train_ds, "stoch")
    trunk = _stack_key(train_ds, "trunk_y").reshape(
        -1, _stack_key(train_ds, "trunk_y").shape[-1]
    )
    target = _stack_key(train_ds, "target").reshape(-1)
    return {
        "stoch_mean": stoch.mean(0),
        "stoch_std": stoch.std(0).clamp_min(1e-6),
        "trunk_mean": trunk.mean(0),
        "trunk_std": trunk.std(0).clamp_min(1e-6),
        "target_mean": target.mean(),
        "target_std": target.std().clamp_min(1e-6),
    }


def apply_norms(ds, stats: dict[str, torch.Tensor]) -> None:
    for i in range(len(ds)):
        item = ds._cache[i]
        item["stoch"] = (item["stoch"] - stats["stoch_mean"]) / stats["stoch_std"]
        item["trunk_y"] = (item["trunk_y"] - stats["trunk_mean"]) / stats["trunk_std"]
        item["target_raw"] = item["target"].clone()
        item["target"] = (item["target"] - stats["target_mean"]) / stats["target_std"]


def fit_and_apply_norms(train_ds, *other_ds) -> dict[str, torch.Tensor]:
    """Z-score stoch, trunk, and target using the training split only."""
    stats = fit_norms(train_ds)
    apply_norms(train_ds, stats)
    for ds in other_ds:
        apply_norms(ds, stats)
    return stats


def stats_from_checkpoint(blob: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Restore train-split z-score tensors saved in a leftover checkpoint."""
    raw = blob.get("stats") or {}
    out: dict[str, torch.Tensor] = {}
    for key, val in raw.items():
        t = val if torch.is_tensor(val) else torch.as_tensor(val)
        out[str(key)] = t.detach().float().clone()
    need = (
        "stoch_mean",
        "stoch_std",
        "trunk_mean",
        "trunk_std",
        "target_mean",
        "target_std",
    )
    missing = [k for k in need if k not in out]
    if missing:
        raise ValueError(f"checkpoint stats missing {missing}")
    return out


def _pad_trunk_stats_to(stats: dict[str, torch.Tensor], trunk_dim: int) -> None:
    """Match ship trunk_mean/std to a widened query (extra Fourier harmonics).

    New columns are mean 0 / std 1 so sin/cos harmonics enter unscaled; the
    checkpoint Linear is zero-padded on the same extra inputs.
    """
    mean = stats["trunk_mean"]
    std = stats["trunk_std"]
    d = int(mean.shape[-1])
    if d == trunk_dim:
        return
    if d > trunk_dim:
        raise ValueError(
            f"checkpoint trunk stats dim {d} exceeds query dim {trunk_dim}"
        )
    extra = trunk_dim - d
    stats["trunk_mean"] = torch.cat([mean, mean.new_zeros(extra)], dim=-1)
    stats["trunk_std"] = torch.cat([std, std.new_ones(extra)], dim=-1)
    print(
        f"[train] padded trunk stats {d} → {trunk_dim} (new harmonics mean=0 std=1)",
        flush=True,
    )


def _pad_stoch_stats_to(stats: dict[str, torch.Tensor], stoch_dim: int) -> None:
    """Match ship stoch_mean/std when the branch layout changes.

    Growing (ξ+CoV → +ACF) pads new channels with mean 0 / std 1. Shrinking
    (legacy20 → xi_cov / xi_field_acf) keeps ξ and CoV and drops rH / aHV /
    ξ_damp. Unmappable widths still raise.
    """
    mean = stats["stoch_mean"]
    std = stats["stoch_std"]
    d = int(mean.shape[-1])
    if d == stoch_dim:
        return
    try:
        src_layout = stoch_layout_from_dim(d)
        dst_layout = stoch_layout_from_dim(stoch_dim)
        stats["stoch_mean"] = remap_stoch_last_dim(mean, stoch_dim, new_fill=0.0)
        stats["stoch_std"] = remap_stoch_last_dim(std, stoch_dim, new_fill=1.0)
    except ValueError as exc:
        raise ValueError(
            f"checkpoint stoch stats dim {d} exceeds branch dim {stoch_dim}"
        ) from exc
    print(
        f"[train] remapped stoch stats {d} → {stoch_dim} "
        f"({src_layout} → {dst_layout}; new channels mean=0 std=1)",
        flush=True,
    )


def apply_checkpoint_stats(stats: dict[str, torch.Tensor], *datasets) -> None:
    trunk_dim = None
    stoch_dim = None
    for ds in datasets:
        cache = getattr(ds, "_cache", None)
        if cache:
            trunk_dim = int(cache[0]["trunk_y"].shape[-1])
            stoch_dim = int(cache[0]["stoch"].shape[-1])
            break
    if trunk_dim is not None:
        _pad_trunk_stats_to(stats, trunk_dim)
    if stoch_dim is not None:
        _pad_stoch_stats_to(stats, stoch_dim)
    for ds in datasets:
        apply_norms(ds, stats)


def domain_item_indices(ds, domain: str) -> list[int]:
    """Local indices into a CombinedResidualDataset for one mix domain."""
    names = getattr(ds, "domain_names_per_item", None)
    if not names:
        return []
    key = str(domain).lower()
    out: list[int] = []
    for i, name in enumerate(names):
        n = str(name).lower()
        if key in ("three_layer", "ood_three_layer"):
            if "three" in n:
                out.append(i)
        elif key in ("dipping", "ood_dipping"):
            if "dip" in n:
                out.append(i)
        elif key == "iid":
            if n.startswith("iid"):
                out.append(i)
    return out


def dataset_freq(ds) -> np.ndarray | None:
    """Hz grid for leftover band metrics. Unwraps ``Subset`` (val OOD slices)."""
    cur = ds
    for _ in range(4):
        if cur is None:
            break
        freq = getattr(cur, "freq_s", None)
        if freq is not None:
            return freq
        cur = getattr(cur, "dataset", None)
    return None


def _mirror_vanilla_fno_into_band2(state: dict[str, Any]) -> dict[str, Any]:
    """Copy ship vanilla `fno` / `proj` weights onto both band2 heads."""
    extra: dict[str, Any] = {}
    for key, val in state.items():
        if ".fno." in key:
            extra[key.replace(".fno.", ".fno_low.", 1)] = val
            extra[key.replace(".fno.", ".fno_high.", 1)] = val
        elif key.startswith("fno."):
            extra["fno_low." + key[4:]] = val
            extra["fno_high." + key[4:]] = val
        elif ".proj." in key and ".proj_high." not in key:
            extra[key.replace(".proj.", ".proj_high.", 1)] = val
        elif key.startswith("proj."):
            extra["proj_high." + key[5:]] = val
    out = dict(state)
    out.update(extra)
    return out


def _compatible_state(
    model: nn.Module, state: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Drop keys whose shapes do not match (modes-up / leftover-head swaps).

    Also maps DeepONetFNO ``base.*`` keys onto an unwrapped kernel GNO so ship
    ``col_enc`` / fuse / trunk can warm-start when FNO-on-R is off.
    """
    current = model.state_dict()
    out: dict[str, Any] = {}
    skipped: list[str] = []

    def _candidates(key: str) -> list[str]:
        names = [key]
        if key.startswith("base."):
            names.append(key[5:])
        else:
            names.append("base." + key)
        return names

    for key, val in state.items():
        for ck in _candidates(str(key)):
            if ck not in current:
                continue
            if tuple(current[ck].shape) == tuple(val.shape):
                out.setdefault(ck, val)
                break
            adapted = None
            key_l = str(ck).lower()
            if "stoch_mlp" in key_l:
                adapted = _adapt_stoch_input_dim(current[ck], val)
            elif "trunk" in key_l:
                adapted = _zero_pad_input_dim(current[ck], val)
            if adapted is not None:
                out.setdefault(ck, adapted)
                break
            skipped.append(str(key))
            break
    return out, skipped


def _zero_pad_input_dim(target: Any, val: Any) -> Any | None:
    """Warm-start a Linear whose input grew (extra trunk harmonics) with zeros.

    The added columns start at zero, so the widened model reproduces the
    checkpoint exactly at init and the new features enter only through training.
    """
    if val.ndim != 2 or target.ndim != 2:
        return None
    if target.shape[0] != val.shape[0] or target.shape[1] <= val.shape[1]:
        return None
    out = torch.zeros_like(target)
    out[:, : val.shape[1]] = val
    return out


def _adapt_stoch_input_dim(target: Any, val: Any) -> Any | None:
    """Warm-start ``stoch_mlp`` when the branch layout is not a prefix pad.

    Prefix copy of a 20-d legacy weight onto 17/18-d would treat rH/aHV as
    CoV/ACF. Named remap keeps ξ + CoV and zero-fills channels the dest adds.
    """
    if val.ndim != 2 or target.ndim != 2:
        return None
    if target.shape[0] != val.shape[0]:
        return None
    if target.shape[1] == val.shape[1]:
        return val
    try:
        return remap_stoch_last_dim(val, int(target.shape[1]), new_fill=0.0)
    except ValueError:
        return _zero_pad_input_dim(target, val)


def _load_init_weights(
    model: nn.Module,
    blob: dict[str, Any],
    *,
    fno_kind: str,
    init_ckpt: Path,
) -> None:
    state = blob["model"]
    if str(fno_kind) == "tf":
        state = {k: v for k, v in state.items() if str(k).startswith("base.")}
    elif str(fno_kind) == "band2":
        state = _mirror_vanilla_fno_into_band2(state)
    filtered, skipped = _compatible_state(model, state)
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(
        f"[train] loaded weights from {init_ckpt} "
        f"missing={len(missing)} unexpected={len(unexpected)} "
        f"shape_skip={len(skipped)}",
        flush=True,
    )
    if missing:
        print(f"[train] init missing keys (first 8): {list(missing)[:8]}", flush=True)
    if skipped:
        print(f"[train] init shape-skip keys (first 8): {skipped[:8]}", flush=True)


def convert_targets_to_log_residual(ds, eps: float = 1e-8) -> None:
    """Replace additive R with log-multiplicative leftover in the in-memory cache."""
    for i in range(len(ds)):
        item = ds._cache[i]
        tf1d = item["tf1d"].clamp_min(eps)
        tf2d = item["tf2d"].clamp_min(eps)
        item["target"] = torch.log(tf2d) - torch.log(tf1d)


def _radial_loss_mod():
    import importlib.util

    path = Path(__file__).resolve().parent.parent / "GIFNO" / "radial_spectral_loss.py"
    spec = importlib.util.spec_from_file_location("_gifno_radial_spectral_loss", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.RadialBinnedSpectralLoss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mode: BranchMode,
    stats: dict[str, torch.Tensor],
    *,
    n_rec: int,
    n_freq: int,
    freq: np.ndarray | None = None,
) -> dict[str, float]:
    """Primary metrics are on signed R (what TF_1D cannot explain).

    TF reconstruction R² is reported only as a secondary diagnostic, together with
    the lazy TF_1D-only baseline so a near-zero residual cannot look successful.
    """
    model.eval()
    y_all, pred_all, tf2d_all, tf1d_all, tf_hat_all = [], [], [], [], []
    tf1d_hat_all: list[np.ndarray] = []
    gate_all: list[np.ndarray] = []
    f0_all: list[np.ndarray] = []
    loss_sum, n_batches = 0.0, 0
    crit = nn.SmoothL1Loss(beta=config.SMOOTH_L1_BETA)
    t_mean = stats["target_mean"].to(device)
    t_std = stats["target_std"].to(device)
    apply_query_freq(model, freq)
    for batch in tqdm(loader, desc="eval", leave=False):
        fields = batch["fields"].to(device)
        stoch = batch["stoch"].to(device)
        trunk_y = batch["trunk_y"].to(device)
        target_n = batch["target"].to(device)
        target = batch["target_raw"].to(device)
        tf1d = batch["tf1d"].to(device)
        tf2d = batch["tf2d"].to(device)
        pred_n = _forward(
            model,
            fields,
            stoch,
            trunk_y,
            mode,
            geom_flags=batch.get("geom_flags"),
            rH=batch.get("rH"),
            query_x=batch.get("query_x"),
            support_x=batch.get("support_x"),
        )
        loss_sum += float(crit(pred_n, target_n).item())
        n_batches += 1
        pred = pred_n * t_std + t_mean
        log_residual = bool(getattr(model, "log_residual", False))
        tf_hat = torch.from_numpy(
            tf_from_residual(
                tf1d.cpu().numpy(),
                pred.cpu().numpy(),
                log_residual=log_residual,
            )
        )
        y_all.append(target.cpu().numpy().ravel())
        pred_all.append(pred.cpu().numpy().ravel())
        tf1d_all.append(tf1d.cpu().numpy().ravel())
        tf2d_all.append(tf2d.cpu().numpy().ravel())
        tf_hat_all.append(tf_hat.numpy().ravel())
        if "f0_tts" in batch:
            f0_all.append(
                np.asarray(batch["f0_tts"].cpu().numpy(), dtype=np.float64).ravel()
            )
        core = gno_core(model)
        hat = getattr(core, "last_tf1d_hat", None)
        if hat is not None:
            tf1d_hat_all.append(hat.detach().cpu().numpy().ravel())
        gate = getattr(model, "last_gate", None)
        if gate is not None:
            gate_all.append(
                gate.detach().cpu().numpy().reshape(gate.shape[0], -1).mean(-1)
            )
    y = np.concatenate(y_all)
    p = np.concatenate(pred_all)
    tf1d = np.concatenate(tf1d_all)
    tf2d = np.concatenate(tf2d_all)
    tf_hat = np.concatenate(tf_hat_all)

    leftover = score_leftover_batch(
        tf1d=tf1d,
        r_true=tf2d - tf1d,
        r_hat=p,
        tf2d=tf2d,
        freq=freq,
        n_rec=n_rec,
        n_freq=n_freq,
        gate=np.concatenate(gate_all) if gate_all else None,
        log_residual=bool(getattr(model, "log_residual", False)),
        f0_calc=np.concatenate(f0_all) if f0_all else None,
    )

    # Lazy baselines: predict R̂=0 (TF̂ = TF_1D only)
    zero = np.zeros_like(y)
    r2_tf_1d_only = flat_r2(tf2d, tf1d)
    r2_tf = flat_r2(tf2d, tf_hat)

    mets = {
        # --- primary: residual learning ---
        "smooth_l1": loss_sum / max(n_batches, 1),
        "r2_R": flat_r2(y, p),
        "rel_l2_R": flat_rel_l2(y, p),
        "pearson_R": flat_pearson(y, p),
        "pearson_R_freq": pearson_across_freq(y, p, n_rec=n_rec, n_freq=n_freq),
        "r2_R_zero": flat_r2(y, zero),  # always ~0 if mean≈0; sanity
        "smooth_l1_R_raw": float(
            np.mean(np.where(np.abs(y) < 1.0, 0.5 * y**2, np.abs(y) - 0.5))
        ),
        "smooth_l1_R_pred": float(
            np.mean(
                np.where(
                    np.abs(y - p) < 1.0,
                    0.5 * (y - p) ** 2,
                    np.abs(y - p) - 0.5,
                )
            )
        ),
        # --- secondary: TF recon (must beat TF_1D-only) ---
        "r2_TF": r2_tf,
        "rel_l2_TF": flat_rel_l2(tf2d, tf_hat),
        "rel_l1_TF": leftover.get("rel_l1_tf_hat", _rel_l1(tf2d, tf_hat)),
        "rel_l1_TF_1d_only": leftover.get("rel_l1_tf_1d", _rel_l1(tf2d, tf1d)),
        "rel_l2_TF_1d_only": flat_rel_l2(tf2d, tf1d),
        "r2_TF_1d_only": r2_tf_1d_only,
        "delta_r2_TF": r2_tf - r2_tf_1d_only,
        "pearson_TF_freq": pearson_across_freq(
            tf2d, tf_hat, n_rec=n_rec, n_freq=n_freq
        ),
        "pearson_TF_1d_only_freq": pearson_across_freq(
            tf2d, tf1d, n_rec=n_rec, n_freq=n_freq
        ),
        "harm_rate": leftover["harm_rate"],
        "fail_soft_mean_abs_rhat": leftover["fail_soft_mean_abs_rhat"],
        "logspec_rel_l2_TF": leftover.get("logspec_rel_l2_tf_hat", 0.0),
        "peak_dlnA_vs_2d": leftover.get("peak_dlnA_vs_2d_mean", float("nan")),
        "peak_df0_hz_vs_2d": leftover.get("peak_df0_hz_vs_2d_mean", float("nan")),
        "harm_rate_small_leftover": leftover.get("harm_rate_small_leftover", 0.0),
        "rel_l2_band_low": leftover.get("rel_l2_band_low_hat", float("nan")),
        "rel_l2_band_mid": leftover.get("rel_l2_band_mid_hat", float("nan")),
        "rel_l2_band_high": leftover.get("rel_l2_band_high_hat", float("nan")),
        "rel_l1_band_high": leftover.get("rel_l1_band_high_hat", float("nan")),
        "residual_tv_hat": leftover.get("residual_tv_hat", float("nan")),
    }
    if tf1d_hat_all:
        h = np.concatenate(tf1d_hat_all)
        mets["pearson_1d_hat"] = flat_pearson(tf1d, h)
        mets["learned_1d_kill"] = float(
            mets["pearson_1d_hat"] < config.LEARNED_1D_PEARSON_KILL
        )
    if leftover.get("gate_mean") is not None:
        mets["gate_mean"] = leftover["gate_mean"]
        mets["gate_mean_small_leftover"] = leftover["gate_mean_small_leftover"]
    return mets


def _rel_l1(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.sum(np.abs(y - p)) / max(np.sum(np.abs(y)), 1e-12))


def _forward(
    model: nn.Module,
    fields: torch.Tensor,
    stoch: torch.Tensor,
    trunk_y: torch.Tensor,
    mode: BranchMode,
    geom_flags: torch.Tensor | None = None,
    rH: torch.Tensor | None = None,
    query_x: torch.Tensor | None = None,
    support_x: torch.Tensor | None = None,
) -> torch.Tensor:
    kwargs: dict[str, torch.Tensor] = {}
    if geom_flags is not None and getattr(model, "accepts_geom_flags", False):
        kwargs["geom_flags"] = geom_flags.to(fields.device)
    if getattr(model, "accepts_query_x", False):
        if query_x is not None:
            kwargs["query_x"] = query_x.to(fields.device)
        if support_x is not None:
            kwargs["support_x"] = support_x.to(fields.device)
    if rH is not None:
        rH = rH.to(fields.device)
    from model import apply_gno_dilation

    apply_gno_dilation(model, rH)
    if mode == "stoch_only":
        return model(None, stoch, trunk_y, **kwargs)
    if mode == "fields_only":
        return model(fields, None, trunk_y, **kwargs)
    return model(fields, stoch, trunk_y, **kwargs)


def _arch_kwargs(
    *,
    residual_fno: bool,
    n_rec: int,
    fno_width: int,
    fno_n_modes: tuple[int, int],
    fno_n_layers: int,
    n_gno_layers: int,
    fno_kind: str = "vanilla",
    physics_tokens: bool = False,
    geom_flag_dim: int = 0,
    learned_1d: bool = False,
    gated: bool = False,
    pod_readout: bool = False,
    log_residual: bool = False,
    boost: bool = False,
    boost_fno_kind: str = "loglo",
    boost_width: int | None = None,
    col_enc_depth_tokens: int = 1,
    n_mscale_trunk: int = 1,
    n_mscale_branch: int = 1,
    mscale_coord_dims: tuple[int, ...] = (0, 3),
    col_enc: str = "conv",
    stoch_inject: str = "mlp",
    fuse_kind: str = "mlp",
    gno_rh_dilate: bool = False,
    kernel_k: int = 2,
    latent_fno: bool = False,
    n_latent: int = 32,
) -> dict[str, Any]:
    return {
        "residual_fno": bool(residual_fno),
        "n_rec": int(n_rec),
        "fno_width": int(fno_width),
        "fno_n_modes": tuple(fno_n_modes),
        "fno_n_layers": int(fno_n_layers),
        "n_gno_layers": int(n_gno_layers),
        "fno_kind": str(fno_kind),
        "physics_tokens": bool(physics_tokens),
        "geom_flag_dim": int(geom_flag_dim),
        "learned_1d": bool(learned_1d),
        "gated": bool(gated),
        "pod_readout": bool(pod_readout),
        "log_residual": bool(log_residual),
        "col_enc_depth_tokens": int(col_enc_depth_tokens),
        "boost": bool(boost),
        "boost_fno_kind": str(boost_fno_kind),
        "boost_width": int(boost_width)
        if boost_width is not None
        else max(8, int(fno_width) // 2),
        "boost_shrink": float(config.BOOST_SHRINK),
        "n_mscale_trunk": int(n_mscale_trunk),
        "n_mscale_branch": int(n_mscale_branch),
        "mscale_coord_dims": tuple(int(x) for x in mscale_coord_dims),
        "col_enc": str(col_enc),
        "stoch_inject": str(stoch_inject),
        "fuse_kind": str(fuse_kind),
        "gno_rh_dilate": bool(gno_rh_dilate),
        "kernel_k": int(kernel_k),
        "latent_fno": bool(latent_fno),
        "n_latent": int(n_latent),
    }


def _peak_band_mask(freq_s: np.ndarray, n_rec: int) -> torch.Tensor | None:
    lo, hi = config.PEAK_BAND_HZ
    band = (np.asarray(freq_s) >= lo) & (np.asarray(freq_s) <= hi)
    if not np.any(band):
        return None
    mask = np.broadcast_to(band[None, :], (n_rec, len(freq_s))).reshape(-1)
    return torch.from_numpy(np.ascontiguousarray(mask))


def _batch_rel_l2(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    if mask is not None:
        m = mask.to(pred.device)
        pred = pred[:, m]
        target = target[:, m]
    diff = torch.linalg.vector_norm(pred - target, dim=-1)
    den = torch.linalg.vector_norm(target, dim=-1).clamp_min(1e-12)
    return (diff / den).mean()


def _flatten_domain_metrics(
    per_domain: dict[str, dict[str, float]], prefix: str = "test"
) -> dict[str, float]:
    out: dict[str, float] = {}
    for dname, mets in per_domain.items():
        for key, val in mets.items():
            if isinstance(val, (int, float)):
                out[f"{prefix}/{dname}/{key}"] = float(val)
    return out


def train_from_datasets(
    *,
    train_ds,
    val_ds,
    extra_tests: dict[str, Any],
    target: TargetName,
    branch_mode: BranchMode,
    trunk_set: TrunkSet,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    run_name: str,
    patience: int | None = None,
    no_early_stop: bool = False,
    field_encoder: FieldEncoderKind = "conv",
    n_freq_train: int = config.N_FREQ_TRAIN,
    n_freq_eval: int = config.N_FREQ_EVAL,
    init_ckpt: Path | None = None,
    serial_tf1d: bool = False,
    use_wandb: bool = True,
    iid_frac: float | None = None,
    aux_tf_rel_l2: float = 0.0,
    aux_peak_band: float = 0.0,
    aux_logspec: float = 0.0,
    aux_dct_below: float = 0.0,
    dct_cutoff: int = DCT_CUTOFF,
    smooth_l1_beta: float | None = None,
    residual_fno: bool = False,
    fno_width: int = config.FNO_WIDTH,
    fno_n_modes: tuple[int, int] = config.FNO_N_MODES,
    fno_n_layers: int = config.FNO_N_LAYERS,
    n_gno_layers: int = config.GNO_N_LAYERS,
    fno_kind: str = "vanilla",
    mix_tag: str | None = None,
    lr_sched_factor: float = config.LR_SCHED_FACTOR,
    lr_sched_patience: int = config.LR_SCHED_PATIENCE,
    lr_sched_min: float = config.LR_SCHED_MIN,
    use_lr_sched: bool = True,
    physics_tokens: bool = False,
    geom_flag_dim: int = 0,
    learned_1d: bool = False,
    gated: bool = False,
    gate_sparsity: float = 0.0,
    leftover_tau: float = config.LEFTOVER_GATE_TAU,
    learned_1d_weight: float = 0.0,
    log_residual: bool = False,
    radial_loss_weight: float = 0.0,
    pod_readout: bool = False,
    pod_n_modes: int = config.POD_NUM_MODES,
    pod_modes: np.ndarray | None = None,
    pod_mean: np.ndarray | None = None,
    mix_train_parts: list | None = None,
    boost_ckpt: Path | None = None,
    boost_fno_kind: str = "loglo",
    boost_width: int | None = None,
    boost_shrink: float = config.BOOST_SHRINK,
    freeze_gno: bool = False,
    freeze_fno: bool = False,
    col_enc_depth_tokens: int = 1,
    val_monitor: str = "smooth_l1",
    freq_sample: str = "log",
    band_weight: float = 0.0,
    aux_logspec_full: float = 0.0,
    logspec_full_floor: float = 0.02,
    n_mscale_trunk: int = 1,
    n_mscale_branch: int = 1,
    mscale_coord_dims: tuple[int, ...] = (0, 3),
    col_enc: str = "conv",
    stoch_inject: str = "mlp",
    fuse_kind: str = "mlp",
    gno_rh_dilate: bool = False,
    encoder_lr: float | None = None,
    kernel_k: int = 2,
    latent_fno: bool = False,
    n_latent: int = 32,
) -> dict[str, Any]:
    """Train on already-built datasets; evaluate extra_tests with train-split norms."""
    from torch.utils.data import DataLoader as _DL

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    device = _device()
    if log_residual:
        convert_targets_to_log_residual(train_ds)
        convert_targets_to_log_residual(val_ds)
        for ds in extra_tests.values():
            convert_targets_to_log_residual(ds)
    init_blob: dict[str, Any] | None = None
    if init_ckpt is not None and not (boost_ckpt is not None):
        init_blob = torch.load(init_ckpt, map_location="cpu", weights_only=False)
    if init_blob is not None and init_blob.get("stats"):
        stats = stats_from_checkpoint(init_blob)
        apply_checkpoint_stats(stats, train_ds, val_ds, *extra_tests.values())
        print(f"[train] reused z-score stats from {init_ckpt}", flush=True)
    else:
        stats = fit_and_apply_norms(train_ds, val_ds, *extra_tests.values())
    sampler: WeightedRandomSampler | None = None
    if iid_frac is not None and hasattr(train_ds, "domain_names_per_item"):
        sampler = iid_resample_sampler(train_ds, float(iid_frac))
    train_loader = _DL(
        train_ds,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0,
    )
    val_loader = _DL(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    n_rec = train_ds.n_rec
    n_freq = len(train_ds.f_idx) if hasattr(train_ds, "f_idx") else n_freq_train
    val_domain_loaders: dict[str, Any] = {}
    for dname in ("iid", "ood_dipping", "ood_three_layer"):
        idx = domain_item_indices(val_ds, dname)
        if idx:
            val_domain_loaders[dname] = _DL(
                Subset(val_ds, idx),
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            )
    monitor = str(val_monitor or "smooth_l1")
    if monitor not in ("smooth_l1", "three_layer"):
        raise ValueError(
            f"val_monitor must be smooth_l1 or three_layer, got {monitor!r}"
        )
    if monitor == "three_layer" and "ood_three_layer" not in val_domain_loaders:
        raise ValueError(
            "val_monitor=three_layer needs a three-layer val slice on val_ds"
        )
    trunk_dim = _trunk_dim(train_ds)
    if pod_readout and pod_modes is None:
        from leftover_pod import fit_pod_from_caches, fit_residual_pod

        if mix_train_parts:
            pod_modes, pod_mean = fit_pod_from_caches(
                mix_train_parts,
                n_modes=pod_n_modes,
                log_residual=log_residual,
            )
        else:
            r_phys = torch.stack([item["target"] for item in train_ds._cache], 0)
            r_phys = r_phys.reshape(r_phys.shape[0], n_rec, -1).numpy()
            pod_modes, pod_mean = fit_residual_pod(r_phys, n_modes=pod_n_modes)
        print(
            f"[train] leftover POD k={pod_modes.shape[1]} grid={pod_modes.shape}",
            flush=True,
        )
    boosting = boost_ckpt is not None
    booster_width = (
        int(boost_width) if boost_width is not None else max(8, int(fno_width) // 2)
    )
    arch_kw = _arch_kwargs(
        residual_fno=residual_fno,
        n_rec=n_rec,
        fno_width=fno_width if not boosting else booster_width,
        fno_n_modes=fno_n_modes,
        fno_n_layers=fno_n_layers,
        n_gno_layers=n_gno_layers,
        fno_kind=fno_kind,
        physics_tokens=physics_tokens,
        geom_flag_dim=geom_flag_dim,
        learned_1d=learned_1d,
        gated=gated,
        pod_readout=pod_readout,
        log_residual=log_residual,
        col_enc_depth_tokens=col_enc_depth_tokens,
        boost=boosting,
        boost_fno_kind=boost_fno_kind,
        boost_width=booster_width,
        n_mscale_trunk=n_mscale_trunk,
        n_mscale_branch=n_mscale_branch,
        mscale_coord_dims=mscale_coord_dims,
        col_enc=col_enc,
        stoch_inject=stoch_inject,
        fuse_kind=fuse_kind,
        gno_rh_dilate=gno_rh_dilate,
        kernel_k=kernel_k,
        latent_fno=latent_fno,
        n_latent=n_latent,
    )
    arch_kw["boost_shrink"] = float(boost_shrink)
    build_kw = dict(arch_kw)
    build_kw["boost"] = False
    model = build_model(
        branch_mode,
        field_channels=config.FIELD_CHANNELS,
        stoch_dim=stoch_dim_from_dataset(train_ds),
        trunk_dim=trunk_dim,
        latent_dim=config.LATENT_DIM,
        field_hidden=config.FIELD_HIDDEN,
        branch_hidden=config.BRANCH_HIDDEN,
        trunk_hidden=config.TRUNK_HIDDEN,
        trunk_layers=config.TRUNK_LAYERS,
        field_encoder=field_encoder,
        pod_modes=pod_modes,
        pod_mean=pod_mean,
        pod_n_modes=pod_n_modes,
        loglo_patch=config.LOGLO_PATCH,
        **build_kw,
    ).to(device)
    if boosting:
        blob = torch.load(boost_ckpt, map_location="cpu", weights_only=False)
        frozen_arch = {
            "field_encoder": blob.get("field_encoder", field_encoder),
            "residual_fno": bool(blob.get("residual_fno", True)),
            "n_rec": int(blob.get("n_rec", n_rec)),
            "fno_width": int(blob.get("fno_width", fno_width)),
            "fno_n_modes": tuple(blob.get("fno_n_modes", fno_n_modes)),
            "fno_n_layers": int(blob.get("fno_n_layers", fno_n_layers)),
            "n_gno_layers": int(blob.get("n_gno_layers", n_gno_layers)),
            "fno_kind": blob.get("fno_kind", "vanilla"),
            "physics_tokens": bool(blob.get("physics_tokens", False)),
            "geom_flag_dim": int(blob.get("geom_flag_dim", 0)),
            "learned_1d": bool(blob.get("learned_1d", False)),
            "gated": bool(blob.get("gated", False)),
            "pod_readout": bool(blob.get("pod_readout", False)),
            "log_residual": bool(blob.get("log_residual", False)),
            "loglo_patch": tuple(blob.get("loglo_patch", config.LOGLO_PATCH)),
            "pod_n_modes": int(blob.get("pod_n_modes", pod_n_modes)),
        }
        frozen = build_model(
            blob.get("branch_mode", branch_mode),
            field_channels=config.FIELD_CHANNELS,
            stoch_dim=stoch_dim_from_dataset(train_ds),
            trunk_dim=trunk_dim,
            latent_dim=config.LATENT_DIM,
            field_hidden=config.FIELD_HIDDEN,
            branch_hidden=config.BRANCH_HIDDEN,
            trunk_hidden=config.TRUNK_HIDDEN,
            trunk_layers=config.TRUNK_LAYERS,
            pod_modes=blob.get("pod_modes"),
            pod_mean=blob.get("pod_mean"),
            boost=False,
            **frozen_arch,
        )
        frozen.load_state_dict(blob["model"])
        booster = model
        model = FrozenBoost(frozen, booster, shrink=boost_shrink).to(device)
        model.log_residual = bool(log_residual)  # type: ignore[attr-defined]
        booster_arch = {
            "field_encoder": field_encoder,
            "residual_fno": bool(residual_fno),
            "n_rec": int(n_rec),
            "fno_width": int(booster_width),
            "fno_n_modes": tuple(fno_n_modes),
            "fno_n_layers": int(fno_n_layers),
            "n_gno_layers": int(n_gno_layers),
            "fno_kind": str(fno_kind),
            "physics_tokens": bool(physics_tokens),
            "geom_flag_dim": int(geom_flag_dim),
            "learned_1d": bool(learned_1d),
            "gated": bool(gated),
            "pod_readout": bool(pod_readout),
            "log_residual": bool(log_residual),
            "loglo_patch": tuple(config.LOGLO_PATCH),
            "pod_n_modes": int(pod_n_modes),
        }
        arch_kw["frozen_arch"] = frozen_arch
        arch_kw["booster_arch"] = booster_arch
        arch_kw["fno_kind"] = str(frozen_arch["fno_kind"])
        arch_kw["fno_width"] = int(frozen_arch["fno_width"])
        arch_kw["boost_fno_kind"] = str(fno_kind)
        arch_kw["boost_width"] = int(booster_width)
        print(f"[train] frozen leftover from {boost_ckpt}", flush=True)
    if init_ckpt is not None and not boosting:
        blob = init_blob or torch.load(
            init_ckpt, map_location="cpu", weights_only=False
        )
        _load_init_weights(model, blob, fno_kind=str(fno_kind), init_ckpt=init_ckpt)
    if freeze_gno:
        n_frozen = freeze_gno_encoder(model)
        print(f"[train] froze GNO encoder params n={n_frozen}", flush=True)
        if n_frozen == 0:
            print(
                "[train] WARNING: --freeze-gno found no col_enc/gno params", flush=True
            )
        if str(field_encoder) == "kernel":
            print(
                "[train] kernel mixer stays trainable (--freeze-gno only froze col_enc)",
                flush=True,
            )
    if freeze_fno:
        n_fno = freeze_fno_head(model)
        print(f"[train] froze FNO leftover head params n={n_fno}", flush=True)
        if n_fno == 0:
            print(
                "[train] WARNING: --freeze-fno found no DeepONetFNO params", flush=True
            )

    enc_lr = encoder_lr
    if enc_lr is None and gno_rh_dilate and not freeze_gno:
        enc_lr = float(lr) * 0.1
    trainable = [p for p in model.parameters() if p.requires_grad]
    if enc_lr is not None and enc_lr > 0 and not freeze_gno:
        core = gno_core(model)
        enc_params: list[nn.Parameter] = []
        enc_ids: set[int] = set()
        for name in ("col_enc", "gno"):
            sub = getattr(core, name, None)
            if sub is None:
                continue
            for p in sub.parameters():
                if p.requires_grad:
                    enc_params.append(p)
                    enc_ids.add(id(p))
        other = [p for p in trainable if id(p) not in enc_ids]
        groups = []
        if other:
            groups.append({"params": other, "lr": lr})
        if enc_params:
            groups.append({"params": enc_params, "lr": float(enc_lr)})
            print(
                f"[train] encoder lr={float(enc_lr):.1e} "
                f"(col_enc/gno n={sum(p.numel() for p in enc_params)})",
                flush=True,
            )
        opt = torch.optim.AdamW(
            groups or trainable,
            lr=lr,
            betas=config.ADAMW_BETAS,
            weight_decay=config.WEIGHT_DECAY,
        )
    else:
        opt = torch.optim.AdamW(
            trainable,
            lr=lr,
            betas=config.ADAMW_BETAS,
            weight_decay=config.WEIGHT_DECAY,
        )
    sched: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None
    if use_lr_sched:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=float(lr_sched_factor),
            patience=int(lr_sched_patience),
            min_lr=float(lr_sched_min),
            threshold=1e-8,
        )
    beta = float(
        smooth_l1_beta if smooth_l1_beta is not None else config.SMOOTH_L1_BETA
    )
    crit = nn.SmoothL1Loss(beta=beta)
    radial_crit = _radial_loss_mod()() if radial_loss_weight > 0 else None
    t_mean = stats["target_mean"].to(device)
    t_std = stats["target_std"].to(device)
    freq_s = getattr(train_ds, "freq_s", None)
    peak_mask = (
        _peak_band_mask(np.asarray(freq_s), n_rec) if freq_s is not None else None
    )
    band_masks = None
    band_scales = None
    if band_weight > 0 and freq_s is not None:
        _fs = np.asarray(freq_s, dtype=float)
        _bands = [config.FREQ_BAND_LOW, config.FREQ_BAND_MID, config.FREQ_BAND_HIGH]
        _rows = [
            (_fs >= lo) & (_fs <= hi if i == len(_bands) - 1 else _fs < hi)
            for i, (lo, hi) in enumerate(_bands)
        ]
        band_masks = torch.from_numpy(np.stack(_rows)).to(device)
    wandb_run = init_wandb(
        run_name,
        {
            "encoder": field_encoder,
            "serial_tf1d": serial_tf1d,
            "n_freq_train": n_freq_train,
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "patience": patience if patience is not None else config.PATIENCE,
            "lr_sched": bool(use_lr_sched),
            "lr_sched_factor": float(lr_sched_factor),
            "lr_sched_patience": int(lr_sched_patience),
            "lr_sched_min": float(lr_sched_min),
            "val_monitor": str(val_monitor),
            "freeze_gno": bool(freeze_gno),
            "freeze_fno": bool(freeze_fno),
            "col_enc_depth_tokens": int(col_enc_depth_tokens),
            "col_enc": str(col_enc),
            "stoch_inject": str(stoch_inject),
            "fuse_kind": str(fuse_kind),
            "gno_rh_dilate": bool(gno_rh_dilate),
            "kernel_k": int(kernel_k),
            "latent_fno": bool(latent_fno),
            "n_latent": int(n_latent),
            "encoder_lr": None if encoder_lr is None else float(encoder_lr),
            "seed": seed,
            "iid_frac": iid_frac,
            "aux_tf_rel_l2": aux_tf_rel_l2,
            "aux_peak_band": aux_peak_band,
            "aux_logspec": aux_logspec,
            "aux_dct_below": aux_dct_below,
            "dct_cutoff": int(dct_cutoff),
            "smooth_l1_beta": beta,
            "freq_sample": str(freq_sample),
            "residual_fno": residual_fno,
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "host": os.environ.get("WANDB_HOST", "laptop"),
            "mix": mix_tag,
            "n_freq": n_freq_train,
            "radial_loss_weight": float(radial_loss_weight),
            "boost_ckpt": str(boost_ckpt) if boost_ckpt else None,
            **arch_kw,
        },
        enabled=use_wandb,
    )
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    patience_budget = int(patience if patience is not None else config.PATIENCE)
    patience_left = patience_budget
    history: list[dict] = []
    ckpt_path = config.CHECKPOINT_DIR / f"{run_name}.pt"
    t0 = time.time()
    epoch_bar = trange(1, epochs + 1, desc=run_name)
    for epoch in epoch_bar:
        apply_query_freq(model, getattr(train_ds, "freq_s", None))
        model.train()
        tr_loss, tr_n = 0.0, 0
        tr_aux, tr_peak = 0.0, 0.0
        for batch in tqdm(train_loader, desc="train", leave=False):
            fields = batch["fields"].to(device)
            stoch = batch["stoch"].to(device)
            trunk_y = batch["trunk_y"].to(device)
            target_t = batch["target"].to(device)
            pred = _forward(
                model,
                fields,
                stoch,
                trunk_y,
                branch_mode,
                geom_flags=batch.get("geom_flags"),
                rH=batch.get("rH"),
                query_x=batch.get("query_x"),
                support_x=batch.get("support_x"),
            )
            loss = crit(pred, target_t)
            if radial_crit is not None:
                pred_raw = pred * t_std + t_mean
                tf1d_b = batch["tf1d"].to(device)
                tf2d_b = batch["tf2d"].to(device)
                if bool(getattr(model, "log_residual", False)):
                    tf_hat_t = tf1d_b * torch.exp(pred_raw.clamp(-20.0, 20.0))
                else:
                    tf_hat_t = tf1d_b + pred_raw
                loss = loss + float(radial_loss_weight) * radial_crit(
                    tf_hat_t.reshape(pred.shape[0], n_rec, n_freq),
                    tf2d_b.reshape(pred.shape[0], n_rec, n_freq),
                )
            if gate_sparsity > 0:
                gate = getattr(model, "last_gate", None)
                if gate is not None:
                    r_phys = batch["tf2d"].to(device) - batch["tf1d"].to(device)
                    tf1d_b = batch["tf1d"].to(device)
                    lev = torch.linalg.vector_norm(
                        r_phys, dim=-1
                    ) / torch.linalg.vector_norm(tf1d_b, dim=-1).clamp_min(1e-12)
                    small = lev < float(leftover_tau)
                    if bool(small.any()):
                        g = gate.reshape(gate.shape[0], -1).mean(dim=-1)
                        loss = loss + float(gate_sparsity) * g[small].mean()
            if learned_1d_weight > 0:
                core = gno_core(model)
                hat = getattr(core, "last_tf1d_hat", None)
                if hat is not None:
                    loss = loss + float(learned_1d_weight) * crit(
                        hat, batch["tf1d"].to(device)
                    )
            if band_weight > 0 and band_masks is not None:
                pr = (pred * t_std + t_mean).reshape(pred.shape[0], n_rec, n_freq)
                rt = (
                    batch["target_raw"].to(device).reshape(pred.shape[0], n_rec, n_freq)
                )
                if band_scales is None:
                    band_scales = torch.stack(
                        [
                            rt[..., band_masks[b]].pow(2).mean().sqrt()
                            for b in range(band_masks.shape[0])
                        ]
                    ).detach()
                loss = loss + float(band_weight) * band_normalized_loss(
                    pr, rt, band_masks, band_scales
                )
            if (
                aux_tf_rel_l2 > 0
                or aux_peak_band > 0
                or aux_logspec > 0
                or aux_logspec_full > 0
            ):
                pred_raw = pred * t_std + t_mean
                tf1d = batch["tf1d"].to(device)
                tf2d = batch["tf2d"].to(device)
                if bool(getattr(model, "log_residual", False)):
                    tf_hat = tf1d * torch.exp(pred_raw.clamp(-20.0, 20.0))
                else:
                    tf_hat = tf1d + pred_raw
                if aux_tf_rel_l2 > 0:
                    aux = _batch_rel_l2(tf_hat, tf2d, None)
                    loss = loss + float(aux_tf_rel_l2) * aux
                    tr_aux += float(aux.item()) * target_t.shape[0]
                if aux_peak_band > 0 and peak_mask is not None:
                    peak = _batch_rel_l2(tf_hat, tf2d, peak_mask)
                    loss = loss + float(aux_peak_band) * peak
                    tr_peak += float(peak.item()) * target_t.shape[0]
                if aux_logspec > 0:
                    ls = trough_safe_logspec_loss(tf_hat, tf2d)
                    loss = loss + float(aux_logspec) * ls
                if aux_logspec_full > 0:
                    lsf = logspec_full_loss(
                        tf_hat.reshape(pred.shape[0], n_rec, n_freq),
                        tf2d.reshape(pred.shape[0], n_rec, n_freq),
                        floor_frac=float(logspec_full_floor),
                    )
                    loss = loss + float(aux_logspec_full) * lsf
            if aux_dct_below > 0:
                pred_raw = pred * t_std + t_mean
                r_true = batch["target_raw"].to(device)
                dct_l = dct_below_cutoff_loss(
                    pred_raw, r_true, n_rec=n_rec, cutoff=int(dct_cutoff)
                )
                loss = loss + float(aux_dct_below) * dct_l
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item()) * target_t.shape[0]
            tr_n += target_t.shape[0]
        val_m = evaluate(
            model,
            val_loader,
            device,
            branch_mode,
            stats,
            n_rec=n_rec,
            n_freq=n_freq,
            freq=dataset_freq(val_ds),
        )
        val_by_domain: dict[str, dict[str, float]] = {}
        for dname, loader in val_domain_loaders.items():
            val_by_domain[dname] = evaluate(
                model,
                loader,
                device,
                branch_mode,
                stats,
                n_rec=n_rec,
                n_freq=n_freq,
                freq=dataset_freq(loader.dataset),
            )
        train_sL1 = tr_loss / max(tr_n, 1)
        if monitor == "three_layer":
            monitor_val = float(val_by_domain["ood_three_layer"]["rel_l2_TF"])
            monitor_name = "val/ood_three_layer/rel_l2_TF"
        else:
            monitor_val = float(val_m["smooth_l1"])
            monitor_name = "val/smooth_l1"
        row = {
            "epoch": epoch,
            "train_smooth_l1": train_sL1,
            "val_monitor": monitor_val,
            **{f"val_{k}": v for k, v in val_m.items()},
        }
        if aux_tf_rel_l2 > 0:
            row["train_tf_rel_l2"] = tr_aux / max(tr_n, 1)
        if aux_peak_band > 0:
            row["train_peak_rel_l2"] = tr_peak / max(tr_n, 1)
        history.append(row)
        improved = monitor_val < best_val - 1e-8
        if improved:
            best_val = monitor_val
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            patience_left = patience_budget
        elif not no_early_stop:
            patience_left -= 1
        postfix = {
            "sL1": f"{train_sL1:.3e}",
            "r2_R": f"{val_m['r2_R']:.3f}",
            "lr": f"{opt.param_groups[0]['lr']:.1e}",
        }
        if monitor == "three_layer":
            postfix["3L"] = f"{monitor_val:.3f}"
        epoch_bar.set_postfix(**postfix)
        payload = {
            "epoch": epoch,
            "train/smooth_l1": train_sL1,
            "lr": opt.param_groups[0]["lr"],
            "val/smooth_l1": val_m["smooth_l1"],
            "val/r2_R": val_m["r2_R"],
            "val/pearson_R_freq": val_m["pearson_R_freq"],
            "val/delta_r2_TF": val_m["delta_r2_TF"],
            "val/rel_l2_TF": val_m["rel_l2_TF"],
            "val/rel_l2_TF_1d_only": val_m["rel_l2_TF_1d_only"],
            "val/monitor": monitor_val,
            "val/best_monitor": best_val,
            "val/best_epoch": best_epoch,
        }
        payload.update(_flatten_domain_metrics(val_by_domain, prefix="val"))
        if "train_tf_rel_l2" in row:
            payload["train/tf_rel_l2"] = row["train_tf_rel_l2"]
        if "train_peak_rel_l2" in row:
            payload["train/peak_rel_l2"] = row["train_peak_rel_l2"]
        log_wandb(wandb_run, payload, step=epoch)
        if sched is not None:
            prev_lr = float(opt.param_groups[0]["lr"])
            sched.step(monitor_val)
            new_lr = float(opt.param_groups[0]["lr"])
            if new_lr < prev_lr * 0.999:
                print(
                    f"[{run_name}] ReduceLROnPlateau {prev_lr:.2e} → {new_lr:.2e} "
                    f"at epoch {epoch} (monitor={monitor_name})",
                    flush=True,
                )
        if not no_early_stop and not improved and patience_left <= 0:
            print(
                f"[{run_name}] early stop at epoch {epoch} "
                f"(best {monitor_name}={best_val:.4e} @ {best_epoch})",
                flush=True,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if isinstance(model, FrozenBoost):
        best_s = float(model.shrink.detach().cpu())
        best_v = float("inf")
        for s in (0.0, 0.25, 0.5, 0.75, 1.0):
            model.select_shrink_from_val(s)
            vm = evaluate(
                model,
                val_loader,
                device,
                branch_mode,
                stats,
                n_rec=n_rec,
                n_freq=n_freq,
                freq=dataset_freq(val_ds),
            )
            if vm["smooth_l1"] < best_v:
                best_v = vm["smooth_l1"]
                best_s = s
        model.select_shrink_from_val(best_s)
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
        print(
            f"[{run_name}] validation shrink={best_s:.2f} val_sL1={best_v:.4e}",
            flush=True,
        )

    per_domain: dict[str, Any] = {}
    for dname, ds in extra_tests.items():
        loader = _DL(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        per_domain[dname] = evaluate(
            model,
            loader,
            device,
            branch_mode,
            stats,
            n_rec=ds.n_rec,
            n_freq=len(ds.f_idx),
            freq=dataset_freq(ds),
        )
        print(
            f"[{run_name}] test[{dname}] r2_R={per_domain[dname]['r2_R']:.3f} "
            f"Δr2_TF={per_domain[dname]['delta_r2_TF']:+.3f} "
            f"rel_l2_TF={per_domain[dname]['rel_l2_TF']:.3f}",
            flush=True,
        )
    tl = per_domain.get("ood_three_layer") or per_domain.get("three_layer")
    three_layer_kill = bool(
        tl is not None
        and float(tl.get("rel_l2_TF", 0.0)) > float(config.THREE_LAYER_KILL_REL_L2)
    )
    if three_layer_kill:
        print(
            f"[{run_name}] KILL three-layer rel L2 "
            f"{tl['rel_l2_TF']:.3f} > {config.THREE_LAYER_KILL_REL_L2}",
            flush=True,
        )
    flat = _flatten_domain_metrics(per_domain)
    log_wandb(wandb_run, flat)
    summary_wandb(
        wandb_run,
        {
            "best_val_smooth_l1": best_val if monitor == "smooth_l1" else None,
            "best_val_monitor": best_val,
            "val_monitor": monitor,
            "best_epoch": best_epoch,
            "epochs_ran": len(history),
            **flat,
        },
    )

    stats_cpu = {k: v.detach().cpu() for k, v in stats.items()}
    torch.save(
        {
            "model": best_state or model.state_dict(),
            "target": target,
            "branch_mode": branch_mode,
            "trunk_set": trunk_set,
            "field_encoder": field_encoder,
            "n_freq_train": int(n_freq_train),
            "n_freq_eval": int(n_freq_eval),
            "serial_tf1d": bool(serial_tf1d),
            "iid_frac": iid_frac,
            "freeze_gno": bool(freeze_gno),
            "freeze_fno": bool(freeze_fno),
            "col_enc_depth_tokens": int(col_enc_depth_tokens),
            "col_enc": str(col_enc),
            "stoch_inject": str(stoch_inject),
            "fuse_kind": str(fuse_kind),
            "val_monitor": monitor,
            "aux_tf_rel_l2": float(aux_tf_rel_l2),
            "aux_peak_band": float(aux_peak_band),
            "aux_logspec": float(aux_logspec),
            "aux_dct_below": float(aux_dct_below),
            "dct_cutoff": int(dct_cutoff),
            "smooth_l1_beta": float(beta),
            "stats": stats_cpu,
            "test_by_domain": per_domain,
            "history": history,
            "best_epoch": int(best_epoch),
            "stoch_layout": str(getattr(train_ds, "stoch_layout", "xi_cov")),
            "stoch_dim": int(stoch_dim_from_dataset(train_ds)),
            "fstar_kind": str(getattr(train_ds, "fstar_kind", "legacy")),
            "x_coord": str(getattr(train_ds, "x_coord", "x_over_lambda")),
            "nom_variant": str(getattr(train_ds, "nom_variant", "default")),
            "query_split": str(getattr(train_ds, "query_split", "all")),
            "support_stride": int(getattr(train_ds, "support_stride", 0) or 0),
            "pod_modes": None if pod_modes is None else np.asarray(pod_modes),
            "pod_mean": None if pod_mean is None else np.asarray(pod_mean),
            "boost_ckpt": str(boost_ckpt) if boost_ckpt else None,
            **arch_kw,
        },
        ckpt_path,
    )
    result = {
        "name": run_name,
        "target": target,
        "branch_mode": branch_mode,
        "trunk_set": trunk_set,
        "field_encoder": field_encoder,
        "serial_tf1d": bool(serial_tf1d),
        "n_freq_train": int(n_freq_train),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "epochs_ran": len(history),
        "seconds": time.time() - t0,
        "checkpoint": str(ckpt_path),
        "init_ckpt": str(init_ckpt) if init_ckpt else None,
        "test_by_domain": per_domain,
        "best_val_smooth_l1": best_val if monitor == "smooth_l1" else None,
        "best_val_monitor": best_val,
        "val_monitor": monitor,
        "freeze_gno": bool(freeze_gno),
        "freeze_fno": bool(freeze_fno),
        "col_enc_depth_tokens": int(col_enc_depth_tokens),
        "col_enc": str(col_enc),
        "stoch_inject": str(stoch_inject),
        "fuse_kind": str(fuse_kind),
        "gno_rh_dilate": bool(gno_rh_dilate),
        "kernel_k": int(kernel_k),
        "latent_fno": bool(latent_fno),
        "n_latent": int(n_latent),
        "encoder_lr": None if encoder_lr is None else float(encoder_lr),
        "best_epoch": int(best_epoch),
        "seed": seed,
        "iid_frac": iid_frac,
        "aux_tf_rel_l2": float(aux_tf_rel_l2),
        "aux_peak_band": float(aux_peak_band),
        "aux_logspec": float(aux_logspec),
        "aux_dct_below": float(aux_dct_below),
        "dct_cutoff": int(dct_cutoff),
        "smooth_l1_beta": float(beta),
        "freq_sample": str(freq_sample),
        "band_weight": float(band_weight),
        "aux_logspec_full": float(aux_logspec_full),
        "trunk_scales": int(getattr(train_ds, "trunk_scales", 1)),
        "stoch_layout": str(getattr(train_ds, "stoch_layout", "xi_cov")),
        "stoch_dim": int(stoch_dim_from_dataset(train_ds)),
        "fstar_kind": str(getattr(train_ds, "fstar_kind", "legacy")),
        "x_coord": str(getattr(train_ds, "x_coord", "x_over_lambda")),
        "nom_variant": str(getattr(train_ds, "nom_variant", "default")),
        "query_split": str(getattr(train_ds, "query_split", "all")),
        "support_stride": int(getattr(train_ds, "support_stride", 0) or 0),
        "three_layer_kill": bool(
            (per_domain.get("ood_three_layer") or {}).get("rel_l2_TF", 0)
            > config.THREE_LAYER_KILL_REL_L2
        ),
        **arch_kw,
    }
    out_json = config.RESULTS_DIR / "arch_train" / f"{run_name}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, default=str))
    print(f"[{run_name}] wrote {out_json}", flush=True)
    finish_wandb(wandb_run)
    return result


def _trunk_dim(ds) -> int:
    return int(ds._cache[0]["trunk_y"].shape[-1])


def train_one(
    *,
    cache_tag: str,
    target: TargetName,
    branch_mode: BranchMode,
    trunk_set: TrunkSet,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    run_name: str | None = None,
    patience: int | None = None,
    no_early_stop: bool = False,
    field_encoder: FieldEncoderKind = "conv",
    n_freq_train: int = config.N_FREQ_TRAIN,
    n_freq_eval: int = config.N_FREQ_EVAL,
    serial_tf1d: bool = False,
    use_wandb: bool = True,
    lr_sched_factor: float = config.LR_SCHED_FACTOR,
    lr_sched_patience: int = config.LR_SCHED_PATIENCE,
    lr_sched_min: float = config.LR_SCHED_MIN,
    use_lr_sched: bool = True,
) -> dict[str, Any]:
    cache_dir = _build_signed_cache(cache_tag)
    meta = np.load(cache_dir / "meta.npz", allow_pickle=True)
    n = len(meta["sample_idx"])
    splits = make_splits(n, seed=seed)
    enc_tag = "" if field_encoder == "conv" else f"_{field_encoder}"
    name = run_name or f"{branch_mode}{enc_tag}_{trunk_set}_{target}_{cache_tag}"
    train_ds = ResidualDeepONetDataset(
        cache_dir,
        splits.train,
        target=target,
        trunk_set=trunk_set,
        n_freq=n_freq_train,
        serial_tf1d=serial_tf1d,
    )
    val_ds = ResidualDeepONetDataset(
        cache_dir,
        splits.val,
        target=target,
        trunk_set=trunk_set,
        n_freq=n_freq_train,
        serial_tf1d=serial_tf1d,
    )
    extra: dict[str, Any] = {
        "test": ResidualDeepONetDataset(
            cache_dir,
            splits.test,
            target=target,
            trunk_set=trunk_set,
            n_freq=n_freq_train,
            serial_tf1d=serial_tf1d,
        )
    }
    if int(n_freq_eval) != int(n_freq_train):
        extra["test_full"] = ResidualDeepONetDataset(
            cache_dir,
            splits.test,
            target=target,
            trunk_set=trunk_set,
            n_freq=n_freq_eval,
            serial_tf1d=serial_tf1d,
        )
    result = train_from_datasets(
        train_ds=train_ds,
        val_ds=val_ds,
        extra_tests=extra,
        target=target,
        branch_mode=branch_mode,
        trunk_set=trunk_set,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        run_name=name,
        patience=patience,
        no_early_stop=no_early_stop,
        field_encoder=field_encoder,
        n_freq_train=n_freq_train,
        n_freq_eval=n_freq_eval,
        serial_tf1d=serial_tf1d,
        use_wandb=use_wandb,
        lr_sched_factor=lr_sched_factor,
        lr_sched_patience=lr_sched_patience,
        lr_sched_min=lr_sched_min,
        use_lr_sched=use_lr_sched,
    )
    result["cache_tag"] = cache_tag
    result["n_samples"] = n
    result["n_test"] = len(splits.test)
    by = result.get("test_by_domain", {})
    result["test_train_freq"] = by.get("test")
    result["test"] = by.get("test_full", by.get("test"))
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-tag", default="n1000_seed42")
    p.add_argument(
        "--target", choices=["R_col", "R_nom"], default=config.DEFAULT_TARGET
    )
    p.add_argument(
        "--branch-mode",
        choices=["single", "multi", "stoch_only", "fields_only"],
        default="single",
    )
    p.add_argument(
        "--trunk-set",
        choices=["fstar", "fstar_fourier", "xL", "full"],
        default="full",
    )
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--seed", type=int, default=config.SEED)
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
    p.add_argument("--no-early-stop", action="store_true")
    p.add_argument(
        "--field-encoder",
        choices=["conv", "resunet", "gno", "attn", "gat", "kernel"],
        default=config.DEFAULT_FIELD_ENCODER,
    )
    p.add_argument(
        "--serial-tf1d",
        action=argparse.BooleanOptionalAction,
        default=config.DEFAULT_SERIAL_TF1D,
        help="Condition R-hat on log(TF_1D) in the trunk (shipped serial operator).",
    )
    p.add_argument(
        "--n-freq-train",
        type=int,
        default=config.N_FREQ_TRAIN,
        help="Log-spaced trunk queries during training (eval is always full grid).",
    )
    p.add_argument(
        "--n-freq-eval",
        type=int,
        default=config.N_FREQ_EVAL,
        help="Frequency bins for reported test metrics (default: full 1000).",
    )
    p.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=config.WANDB_DEFAULT,
        help="Log train/val/test metrics to Weights & Biases.",
    )
    args = p.parse_args()
    train_one(
        cache_tag=args.cache_tag,
        target=args.target,  # type: ignore[arg-type]
        branch_mode=args.branch_mode,  # type: ignore[arg-type]
        trunk_set=args.trunk_set,  # type: ignore[arg-type]
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        patience=args.patience,
        no_early_stop=args.no_early_stop,
        field_encoder=args.field_encoder,  # type: ignore[arg-type]
        n_freq_train=args.n_freq_train,
        n_freq_eval=args.n_freq_eval,
        serial_tf1d=args.serial_tf1d,
        use_wandb=args.wandb,
        lr_sched_factor=args.lr_sched_factor,
        lr_sched_patience=args.lr_sched_patience,
        lr_sched_min=args.lr_sched_min,
        use_lr_sched=args.lr_sched,
    )


if __name__ == "__main__":
    main()
