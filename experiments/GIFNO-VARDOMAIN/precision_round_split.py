#!/usr/bin/env python3
"""Separate weight rounding from input rounding in the last branch linear.

The bfloat16 gap on these weights is that layer. This runs it in float32
three ways, on the float32 hidden state the rest of the network produced:

  weight   W' = W.to(bfloat16).to(float32), bias rounded the same way, then W' · x
  input    W · round(x), with round(x) = x.to(bfloat16).to(float32)
  both     rounded weights and rounded inputs

``both`` is accumulated in float32 and, as a check, as a real bfloat16 gemm.
The gemm is the autocast matmul that moved run 3848 at 2.5 Hz from 2.73 to 4.00.

The center set is the test-split center station of every horizontal and dipping
record at 500 m, the population of the published harm rate.

    python precision_round_split.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from data import FAMILY_NAME, Cache, WindowSet, eval_records  # noqa: E402
from encoding import CH_MASK  # noqa: E402
from model import sample_grid  # noqa: E402
from precision_diagnose import CHECKPOINTS, encoder_call, g_stats, load_raw, window  # noqa: E402

OUT = HERE / "precision"
NAMES = ("float32", "input_rounded", "weight_rounded", "both_rounded", "bf16_gemm")


def round_bf16(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(dtype=torch.bfloat16).to(dtype=torch.float32)


def hidden_after_first_linear(model, x: torch.Tensor) -> torch.Tensor:
    """Float32 stem, encoder, and first branch linear. Shape (B, width, 256)."""
    h = model.lift(x.index_select(1, model.lift_idx))
    h = model.collapse(h, x[:, CH_MASK])
    h = model.film(model.expand(h), model.site_groups(x))
    xg, xl = encoder_call(model, h, False)
    grid = torch.cat([xg[:, :, 0, :], xl[:, :, 0, :]], dim=1)
    return model.branch[1](model.branch[0](grid.transpose(1, 2).float()))


def last_linear_variants(hidden: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> dict[str, torch.Tensor]:
    """Three float32 matmuls plus the bfloat16 gemm of the same layer."""
    weight_r = round_bf16(weight)
    bias_r = round_bf16(bias)
    hidden_r = round_bf16(hidden)
    return {
        "float32": F.linear(hidden, weight, bias),
        "input_rounded": F.linear(hidden_r, weight, bias),
        "weight_rounded": F.linear(hidden, weight_r, bias_r),
        "weight_rounded_bias_kept": F.linear(hidden, weight_r, bias),
        "both_rounded": F.linear(hidden_r, weight_r, bias_r),
        "bf16_gemm": F.linear(hidden.to(dtype=torch.bfloat16), weight.to(dtype=torch.bfloat16), bias.to(dtype=torch.bfloat16)).float(),
    }


def center_index(valid: torch.Tensor) -> torch.Tensor:
    rows = valid.detach().cpu().numpy()
    return torch.tensor([int(np.flatnonzero(row)[np.flatnonzero(row).size // 2]) for row in rows], dtype=torch.long)


def g_center(coeff_grid: torch.Tensor, query: torch.Tensor, centers: torch.Tensor, model) -> torch.Tensor:
    coeff = sample_grid(coeff_grid.transpose(1, 2), query)
    g = model.pod_mean + torch.einsum("bkm,mf->bkf", coeff, model.pod_modes)
    batch = torch.arange(g.shape[0], device=g.device)
    return g[batch, centers.to(g.device)]


def pearson_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    denom = np.sqrt((a * a).sum(-1) * (b * b).sum(-1))
    return (a * b).sum(-1) / np.maximum(denom, 1e-12)


def _mean_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().mean())


def summarize_family(rows: list[dict], freq: np.ndarray) -> dict:
    if not rows:
        return {"n": 0}
    out: dict = {"n": len(rows)}
    g = {name: np.stack([row[name] for row in rows]) for name in NAMES}
    base = g["float32"]
    both = g["both_rounded"]
    gap = np.abs(both - base)
    out["mean_abs_g_vs_float32"] = {name: float(np.mean(np.abs(g[name] - base))) for name in NAMES}
    out["mean_abs_g_vs_both"] = {name: float(np.mean(np.abs(g[name] - both))) for name in NAMES}
    out["gap_closed_toward_both"] = {
        name: float(1.0 - np.mean(np.abs(g[name] - both)) / max(float(gap.mean()), 1e-12)) for name in NAMES
    }
    k = int(np.argmin(np.abs(freq - 2.5)))
    out["mean_abs_tf_diff_at_2p5_vs_float32"] = {
        name: float(np.mean(np.abs(np.exp(np.stack([row["prior"] for row in rows])[:, k] + g[name][:, k]) - np.exp(np.stack([row["prior"] for row in rows])[:, k] + base[:, k]))))
        for name in NAMES
    }
    prior = np.stack([row["prior"] for row in rows])
    true = np.stack([row["true"] for row in rows])
    column = np.exp(prior)
    truth = np.exp(true)
    pear_col = pearson_rows(column, truth)
    harm = {}
    for name in NAMES:
        model_tf = np.exp(prior + g[name])
        pear = pearson_rows(model_tf, truth)
        harm[name] = float(np.mean(pear < pear_col))
    out["pearson_harm_rate"] = harm
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    cache = Cache(REPO / "data" / "vardomain" / "cache")
    freq = np.asarray(cache.freq, dtype=np.float64)
    model, cfg = load_raw(CHECKPOINTS["stage2"], device)
    lin2 = model.branch[-1]
    weight = lin2.weight.detach()
    bias = lin2.bias.detach()
    weight_delta = (weight - round_bf16(weight)).float()
    report: dict = {
        "layer": "branch[-1]",
        "in_features": int(lin2.in_features),
        "out_features": int(lin2.out_features),
        "weight_rounding": {
            "max_abs_delta": float(weight_delta.abs().max()),
            "rms_delta": float(torch.sqrt((weight_delta ** 2).mean())),
            "fraction_changed": float((weight_delta != 0).float().mean()),
            "bias_max_abs_delta": float((bias - round_bf16(bias)).abs().max()),
        },
        "target": cfg.get("target"),
        "epoch": None,
    }

    x, query, prior, mid, _index = window(cache, "iid", 3848)
    x = x.to(device)
    query = query.to(device)
    with torch.no_grad():
        hidden = hidden_after_first_linear(model, x)
        variants = last_linear_variants(hidden, weight, bias)
        centers = torch.tensor([mid], device=device)
        spectra = {name: g_center(coeff, query, centers, model)[0].float().cpu().numpy() for name, coeff in variants.items()}
    detail = {}
    for name, g in spectra.items():
        stats = g_stats(g, prior, freq)
        detail[name] = {"tf_at_2p5": stats["tf_at_2p5"], "g_at_2p5": stats["g_at_2p5"], "mean_abs_g": stats["mean_abs_g"]}
    base_g = spectra["float32"]
    both_g = spectra["both_rounded"]
    detail["mean_abs_g_vs_float32"] = {name: float(np.mean(np.abs(g - base_g))) for name, g in spectra.items()}
    detail["mean_abs_g_vs_both"] = {name: float(np.mean(np.abs(g - both_g))) for name, g in spectra.items()}
    detail["mean_abs_g_vs_bf16_gemm"] = {name: float(np.mean(np.abs(g - spectra["bf16_gemm"]))) for name, g in spectra.items()}
    report["run_3848"] = detail
    print(json.dumps(report["run_3848"], indent=2), flush=True)

    records = [
        row
        for row in eval_records(cache, "test", families=("iid", "ood_dipping"), positions=1)
        if int(row["width"]) == 500
    ]
    loader = DataLoader(
        WindowSet(cache, "metric", records, stage=0, seed=0, mirror=False),
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )
    by_family: dict[str, list[dict]] = {}
    with torch.no_grad():
        for batch_x, batch_query, batch_tf, batch_prior, batch_valid, batch_info, _xa in loader:
            batch_x = batch_x.to(device)
            batch_query = batch_query.to(device)
            hidden = hidden_after_first_linear(model, batch_x)
            variants = last_linear_variants(hidden, weight, bias)
            centers = center_index(batch_valid)
            gathered = {name: g_center(coeff, batch_query, centers, model).float().cpu().numpy() for name, coeff in variants.items()}
            centers_np = centers.numpy()
            info = batch_info.numpy()
            tf = batch_tf.numpy()
            prior_b = batch_prior.numpy()
            for i in range(info.shape[0]):
                k = int(centers_np[i])
                family = FAMILY_NAME.get(int(info[i, 4]), "section")
                by_family.setdefault(family, []).append(
                    {
                        "run": int(info[i, 1]),
                        "prior": prior_b[i, k].astype(np.float64),
                        "true": tf[i, k].astype(np.float64),
                        **{name: gathered[name][i].astype(np.float64) for name in NAMES},
                    }
                )
            print(f"centers {sum(len(v) for v in by_family.values())}/{len(records)}", flush=True)

    pooled = [row for rows in by_family.values() for row in rows]
    report["center_set"] = {
        "split": "test",
        "width_m": 500,
        "families": summarize_family(pooled, freq) | {"by_family": {name: summarize_family(rows, freq) for name, rows in by_family.items()}},
    }
    # The per-record spectra are not part of the summary.
    (OUT / "round_split.json").write_text(json.dumps(report, indent=2) + "\n")
    _plot(report, freq)
    print(json.dumps(report["center_set"]["families"]["pearson_harm_rate"], indent=2), flush=True)
    print(json.dumps(report["center_set"]["families"]["mean_abs_g_vs_float32"], indent=2), flush=True)
    print(f"wrote {OUT / 'round_split.json'}")


def _plot(report: dict, freq: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    order = ["float32", "input_rounded", "weight_rounded", "both_rounded", "bf16_gemm"]
    labels = ["float32", "rounded inputs", "rounded weights", "both rounded", "bfloat16 gemm"]
    vals = [report["run_3848"][name]["tf_at_2p5"] for name in order]
    colors = ["#4C72B0", "#4C72B0", "#D55E00", "#D55E00", "#D55E00"]
    axes[0].barh(labels[::-1], vals[::-1], color=colors[::-1])
    axes[0].set_xlabel("|TF| at 2.5 Hz")
    axes[0].set_title("Run 3848, last linear only")
    gap = report["center_set"]["families"]["mean_abs_g_vs_float32"]
    axes[1].barh(labels[::-1], [gap[name] for name in order][::-1], color=colors[::-1])
    axes[1].set_xlabel(r"mean $|\Delta g|$ vs float32")
    axes[1].set_title("Test center stations")
    fig.tight_layout()
    fig.savefig(OUT / "round_split.png", dpi=160)
    plt.close(fig)
    del freq


if __name__ == "__main__":
    main()
