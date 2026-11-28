#!/usr/bin/env python3
"""The bfloat16 gap is the station coordinate, not the last-linear matmul.

``sample_grid`` casts the query to the coefficient dtype. Under autocast the
coefficient grid is bfloat16, and 1250 m is not a bfloat16 number: it becomes
1248 m. This script reads both columns from one float32 coefficient grid.

    python precision_query_shift.py
"""

from __future__ import annotations

import argparse
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
from encoding import CONTEXT_M  # noqa: E402
from model import sample_grid  # noqa: E402
from spectral_layers import lateral_mode_mask  # noqa: E402
from precision_diagnose import CHECKPOINTS, g_stats, load_raw, stem_encoder_head, window  # noqa: E402
from precision_round_split import center_index, hidden_after_first_linear, pearson_rows  # noqa: E402

OUT = HERE / "precision"


def coefficients(model, hidden: torch.Tensor) -> torch.Tensor:
    lin2 = model.branch[-1]
    return F.linear(hidden, lin2.weight, lin2.bias)


def g_of(model, coeff: torch.Tensor, query: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    sampled = sample_grid(coeff.transpose(1, 2), query)
    g = model.pod_mean + torch.einsum("bkm,mf->bkf", sampled, model.pod_modes)
    batch = torch.arange(g.shape[0], device=g.device)
    return g[batch, centers.to(g.device)]


def harm(g: np.ndarray, prior: np.ndarray, true: np.ndarray) -> float:
    pear_model = pearson_rows(np.exp(prior + g), np.exp(true))
    pear_column = pearson_rows(np.exp(prior), np.exp(true))
    return float(np.mean(pear_model < pear_column))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    cache = Cache(REPO / "data" / "vardomain" / "cache")
    freq = np.asarray(cache.freq, dtype=np.float64)
    model, _cfg = load_raw(CHECKPOINTS["stage2"], device)
    k25 = int(np.argmin(np.abs(freq - 2.5)))

    x, query, prior, mid, _index = window(cache, "iid", 3848)
    x, query = x.to(device), query.to(device)
    with torch.no_grad():
        coeff = coefficients(model, hidden_after_first_linear(model, x))
        centers = torch.tensor([mid], device=device)
        g_true = g_of(model, coeff, query, centers)[0]
        g_shift = g_of(model, coeff, query.to(dtype=torch.bfloat16).float(), centers)[0]
        y_bf, _c, _h = stem_encoder_head(model, x, query, stem=True, encoder=True, head=True)
        grid = model.pod_mean + torch.einsum("bwm,mf->bwf", coeff, model.pod_modes)
    q = float(query[0, mid])
    q_bf = float(query[0, mid].to(dtype=torch.bfloat16).float())
    lo, hi = int(q) - 20, int(q) + 21
    profile = grid[0, lo:hi, k25].float().cpu().numpy()
    report = {
        "run_3848": {
            "query_m": q,
            "query_bfloat16_m": q_bf,
            "shift_m": q_bf - q,
            "tf_at_true_column": g_stats(g_true.float().cpu().numpy(), prior, freq)["tf_at_2p5"],
            "tf_at_rounded_column": g_stats(g_shift.float().cpu().numpy(), prior, freq)["tf_at_2p5"],
            "tf_full_bfloat16_forward": g_stats(y_bf[0, mid].float().cpu().numpy(), prior, freq)["tf_at_2p5"],
            "mean_abs_g_between_columns": float((g_true - g_shift).float().abs().mean()),
        }
    }
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
    stored: dict[str, list] = {name: [] for name in ("family", "query", "rounded", "g_true", "g_shift", "prior", "true")}
    with torch.no_grad():
        for batch_x, batch_query, batch_tf, batch_prior, batch_valid, batch_info, _xa in loader:
            batch_x = batch_x.to(device)
            batch_query = batch_query.to(device)
            coeff = coefficients(model, hidden_after_first_linear(model, batch_x))
            centers = center_index(batch_valid)
            g0 = g_of(model, coeff, batch_query, centers).float().cpu().numpy()
            g1 = g_of(model, coeff, batch_query.to(dtype=torch.bfloat16).float(), centers).float().cpu().numpy()
            centers_np = centers.numpy()
            info = batch_info.numpy()
            for i in range(info.shape[0]):
                k = int(centers_np[i])
                stored["family"].append(FAMILY_NAME.get(int(info[i, 4]), "section"))
                stored["query"].append(float(batch_query[i, k]))
                stored["rounded"].append(float(batch_query[i, k].to(dtype=torch.bfloat16).float()))
                stored["g_true"].append(g0[i])
                stored["g_shift"].append(g1[i])
                stored["prior"].append(batch_prior[i, k].numpy().astype(np.float64))
                stored["true"].append(batch_tf[i, k].numpy().astype(np.float64))
    g_true = np.stack(stored["g_true"])
    g_shift = np.stack(stored["g_shift"])
    prior_a = np.stack(stored["prior"])
    true_a = np.stack(stored["true"])
    query_a = np.asarray(stored["query"])
    rounded_a = np.asarray(stored["rounded"])
    family = np.asarray(stored["family"])
    summary = {
        "n": int(family.size),
        "unique_query_m": sorted(set(np.round(query_a, 3))),
        "unique_rounded_m": sorted(set(np.round(rounded_a, 3))),
        "n_shifted": int(np.sum(np.abs(query_a - rounded_a) > 1e-6)),
        "mean_abs_g": float(np.mean(np.abs(g_true - g_shift))),
        "pearson_harm_true_column": harm(g_true, prior_a, true_a),
        "pearson_harm_rounded_column": harm(g_shift, prior_a, true_a),
        "by_family": {},
    }
    for name in sorted(set(family)):
        mask = family == name
        summary["by_family"][name] = {
            "n": int(mask.sum()),
            "mean_abs_g": float(np.mean(np.abs(g_true[mask] - g_shift[mask]))),
            "pearson_harm_true_column": harm(g_true[mask], prior_a[mask], true_a[mask]),
            "pearson_harm_rounded_column": harm(g_shift[mask], prior_a[mask], true_a[mask]),
        }
    report["center_set"] = summary
    (OUT / "query_shift.json").write_text(json.dumps(report, indent=2) + "\n")
    _plot(report, np.arange(lo, hi), np.exp(prior[k25] + profile), q, q_bf)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {OUT / 'query_shift.json'}")


def _plot(report: dict, columns: np.ndarray, tf: np.ndarray, query: float, rounded: float) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(columns, tf, color="#4C72B0", lw=1.4)
    ax.axvline(query, color="#4C72B0", ls="--", lw=0.9, label=f"float32 reads {query:.0f} m")
    ax.axvline(rounded, color="#D55E00", ls="--", lw=0.9, label=f"bfloat16 reads {rounded:.0f} m")
    ax.set_xlabel("Canvas column (m)")
    ax.set_ylabel("|TF| at 2.5 Hz")
    ax.set_title("Run 3848, one float32 coefficient grid")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "query_shift.png", dpi=160)
    plt.close(fig)


def power_at_period(series: np.ndarray, period_m: float = 4.0, tol: float = 0.2) -> float:
    """Power of ``series`` (one sample per metre) in a band around ``period_m``."""
    series = np.asarray(series, dtype=np.float64)
    series = series - series.mean()
    power = np.abs(np.fft.rfft(series)) ** 2
    freq = np.fft.rfftfreq(series.size, d=1.0)
    target = 1.0 / period_m
    band = np.abs(freq - target) <= tol * target
    if not np.any(band):
        return 0.0
    return float(power[band].sum())


def dominant_period(series: np.ndarray) -> float:
    series = np.asarray(series, dtype=np.float64)
    series = series - series.mean()
    power = np.abs(np.fft.rfft(series)) ** 2
    freq = np.fft.rfftfreq(series.size, d=1.0)
    power[0] = 0.0
    k = int(np.argmax(power))
    if freq[k] <= 0:
        return float("inf")
    return float(1.0 / freq[k])


def choose_carrier(power: dict[str, float]) -> str:
    """Which ablation removes the 4 m peak.

    ``grid`` is the control: low-passing the coefficient field. If neither the
    HFP residual nor the local patch accounts for most of that drop, both
    operators are cut.
    """
    base = float(power["baseline"])
    drops = {name: base - float(power[name]) for name in ("hfp", "local", "grid")}
    best = "hfp" if drops["hfp"] >= drops["local"] else "local"
    if drops["grid"] > 0 and drops[best] < 0.25 * drops["grid"]:
        return "both"
    return best


def carrier_flags(carrier: str) -> dict[str, float]:
    if carrier == "hfp":
        return {"hfp_kernel": 10, "hfp_stride": 10, "local_min_wavelength": 0.0}
    if carrier == "local":
        return {"hfp_kernel": 4, "hfp_stride": 4, "local_min_wavelength": 10.0}
    return {"hfp_kernel": 10, "hfp_stride": 10, "local_min_wavelength": 10.0}


def _interior(model, x: torch.Tensor, width: int, k25: int) -> np.ndarray:
    cols = torch.arange(CONTEXT_M, CONTEXT_M + width, device=x.device, dtype=torch.float32)
    with torch.no_grad():
        g = model(x, cols.unsqueeze(0))[0, :, k25]
    return g.float().cpu().numpy()


def _lowpass(field: torch.Tensor, min_wavelength: float) -> torch.Tensor:
    spec = torch.fft.rfft(field.float(), dim=-1)
    width = field.shape[-1]
    k = torch.arange(spec.shape[-1], device=spec.device)
    keep = (k == 0) | ((width / k.clamp(min=1).float()) >= min_wavelength)
    return torch.fft.irfft(spec * keep.to(dtype=spec.real.dtype), n=width, dim=-1)


def periodogram_main() -> None:
    """Dense g(x) at 2.5 Hz, then HFP, local-patch, and coefficient-grid ablations."""
    ap = argparse.ArgumentParser(description="Find which lateral path carries the 4 m period.")
    ap.add_argument("--periodogram", action="store_true")
    ap.add_argument("--checkpoint", type=Path, default=CHECKPOINTS["stage2"])
    ap.add_argument("--cache", type=Path, default=REPO / "data" / "vardomain" / "cache")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--limit", type=int, default=4)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = Cache(args.cache)
    freq = np.asarray(cache.freq, dtype=np.float64)
    k25 = int(np.argmin(np.abs(freq - 2.5)))
    model, _cfg = load_raw(args.checkpoint, device)
    records = [
        row for row in eval_records(cache, "test", families=("iid",), positions=1) if int(row["width"]) == 500
    ][: args.limit]
    profiles = {"baseline": [], "hfp": [], "local": [], "grid": []}
    loader = DataLoader(
        WindowSet(cache, "metric", records, stage=0, seed=0, mirror=False),
        batch_size=1, shuffle=False, num_workers=0,
    )
    saved_masks = [(layer, layer.mode_mask.detach().clone()) for layer in model.encoder.local_layers]
    for batch_x, _query, _tf, _prior, _valid, info, _xa in loader:
        batch_x = batch_x.to(device)
        width = int(info[0, 2])
        profiles["baseline"].append(_interior(model, batch_x, width, k25))
        model.encoder.hfp.ablate = True
        profiles["hfp"].append(_interior(model, batch_x, width, k25))
        model.encoder.hfp.ablate = False
        for layer, _mask in saved_masks:
            mask = lateral_mode_mask(int(layer.mode_mask.numel()), int(layer.pw), 10.0)
            layer.mode_mask.copy_(mask.to(device=layer.mode_mask.device))
        profiles["local"].append(_interior(model, batch_x, width, k25))
        for layer, mask in saved_masks:
            layer.mode_mask.copy_(mask)
        cols = torch.arange(CONTEXT_M, CONTEXT_M + width, device=device, dtype=torch.float32)
        with torch.no_grad():
            coeff, latent, _recon = model.encode(batch_x)
            g = model.readout(_lowpass(coeff, 10.0), latent, cols.unsqueeze(0))
        profiles["grid"].append(g[0, :, k25].float().cpu().numpy())
    power = {name: float(np.mean([power_at_period(row) for row in rows])) for name, rows in profiles.items()}
    periods = {name: float(np.median([dominant_period(row) for row in rows])) for name, rows in profiles.items()}
    carrier = choose_carrier(power)
    report = {
        "checkpoint": str(args.checkpoint),
        "n": len(profiles["baseline"]),
        "carrier": carrier,
        "power_at_4m": power,
        "dominant_period_m": periods,
        **carrier_flags(carrier),
    }
    path = args.out / "lateral_periodogram.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    series = profiles["baseline"][0]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(np.arange(series.size), series, color="#4C72B0", lw=1.0)
    ax.set_xlabel("Metres from the interior origin")
    ax.set_ylabel("g at 2.5 Hz")
    ax.set_title(f"Dense query, carrier {carrier}")
    fig.tight_layout()
    fig.savefig(args.out / "lateral_periodogram.png", dpi=160)
    plt.close(fig)
    print(json.dumps(report, indent=2), flush=True)
    print(f"wrote {path}")


if __name__ == "__main__":
    if "--periodogram" in sys.argv:
        periodogram_main()
    else:
        main()
