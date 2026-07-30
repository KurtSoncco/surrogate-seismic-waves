#!/usr/bin/env python3
"""Re-score Response_Variability: Haskell-only vs gated-delta checkpoint vs GIFNO.

Compares center-recorder |TF| to OpenSees-2D for all 64 sobol × RF seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

_EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXP))
import config

config.setup_import_paths()

from context_features import (  # noqa: E402
    bedrock_interface_depth,
    dip_field_broadcast,
    impedance_gradient_field,
    interface_dip,
    residual_gate_scalar,
    stack_delta_input_channels,
)
from haskell_baseline import haskell_af_within  # noqa: E402
from model import create_model  # noqa: E402
from rv_dataset import (  # noqa: E402
    _coord_grids,
    _normalize_vs,
    _normalize_zeta,
    _pad_depth,
    build_rv_index_lookup,
)

_EPS = 1e-12


def _rel_l2(pred, true):
    pred = np.asarray(pred, float).ravel()
    true = np.asarray(true, float).ravel()
    return float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + _EPS))


def _pearson(a, b):
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    if a.size < 2 or np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _stats(xs):
    v = np.asarray(xs, float)
    v = v[np.isfinite(v)]
    return {
        "n": int(v.size),
        "mean": float(v.mean()) if v.size else float("nan"),
        "median": float(np.median(v)) if v.size else float("nan"),
        "p10": float(np.percentile(v, 10)) if v.size else float("nan"),
        "p90": float(np.percentile(v, 90)) if v.size else float("nan"),
    }


def _prepare_context(vs_full, zeta_full, vs2, H, dz, dx):
    i0, i1 = config.X_SLICE_START, config.X_SLICE_END
    vs = vs_full[:, i0:i1]
    zeta = zeta_full[:, i0:i1]
    nz, nx = vs.shape
    z_bed = bedrock_interface_depth(vs, vs_rock=vs2, dz=dz)
    dip = interface_dip(z_bed, dx=dx)
    dip_rms = float(np.sqrt(np.mean(dip**2)))
    imp_g = impedance_gradient_field(vs, rho=config.DEFAULT_RHO, dx=dx)
    dip_2d = dip_field_broadcast(dip, nz)
    vs_pad = _pad_depth(vs, config.NZ_MAX)
    zeta_pad = _pad_depth(zeta, config.NZ_MAX)
    dip_pad = _pad_depth(dip_2d, config.NZ_MAX)
    imp_pad = _pad_depth(imp_g, config.NZ_MAX)
    x_coord, z_coord = _coord_grids(nz, nx, config.NZ_MAX, nz * dz)
    x_in = stack_delta_input_channels(
        _normalize_vs(vs_pad),
        _normalize_zeta(zeta_pad, nz),
        x_coord,
        z_coord,
        dip_pad,
        imp_pad,
    )
    return vs, zeta, x_in, dip_rms, nz


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5-dir", type=Path, default=config.RV_H5_DIR)
    p.add_argument("--checkpoint", type=Path, default=config.MODEL_SAVE_PATH)
    p.add_argument("--out-dir", type=Path, default=config.RESULTS_SAVE_DIR / "rv_rescore")
    p.add_argument("--max-seeds", type=int, default=None)
    p.add_argument("--haskell-only", action="store_true")
    args = p.parse_args()

    sys.path.insert(0, str(config.RV_ROOT))
    os.environ.pop("RV_SMOKE", None)
    from manifest import active_rf_seeds, active_sobol_count  # noqa: WPS433

    lookup = build_rv_index_lookup()
    seeds = active_rf_seeds()
    if args.max_seeds:
        seeds = seeds[: args.max_seeds]

    device = config.DEVICE
    model = None
    if not args.haskell_only and args.checkpoint.is_file():
        model = create_model().to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.eval()
        print(f"Loaded checkpoint {args.checkpoint}")
    else:
        print("Haskell-only mode (no residual checkpoint)")

    center_full = config.BC_WIDTH + config.NX // 2
    rec = config.recorder_x_indices()
    center_strip = int(rec[len(rec) // 2])

    rows = []
    buckets = {
        "haskell_rel": [],
        "haskell_pear": [],
        "delta_rel": [],
        "delta_pear": [],
        "gifno_rel": [],
        "gifno_pear": [],
    }

    # Optional GIFNO grf_2d comparison
    sys.path.insert(0, str(config.RV_ROOT))
    from manifest import _hallal_block_size, index_to_params, total_combinations  # noqa: WPS433

    hallal0 = _hallal_block_size(active_sobol_count())
    grf_lookup = {}
    for i in range(hallal0, total_combinations()):
        pp = index_to_params(i)
        if pp.method == "grf_2d":
            grf_lookup[(pp.sobol_id, pp.seed)] = i

    for sid in range(active_sobol_count()):
        for seed in seeds:
            io = lookup.get((sid, "opensees_2d", seed))
            if io is None:
                continue
            path = args.h5_dir / f"run_{io}.h5"
            if not path.is_file():
                continue
            with h5py.File(path, "r") as f:
                vs_full = f["Vs_field"][:]
                zeta_full = f["Damping_zeta"][:]
                dz = float(f["grid"].attrs.get("dz", 1.0))
                dx = float(f["grid"].attrs.get("dx", 1.0))
                vs2 = float(f["params"].attrs["Vs2"])
                H = float(f["params"].attrs["H"])
                cov = float(f["params"].attrs["CoV"])
                af_ops = f["transfer_function"]["AF"][:].astype(np.float64)
                freq = f["transfer_function"]["freq"][:].astype(np.float64)

            soil_nz = max(1, min(int(round(H / dz)), vs_full.shape[0] - 1))
            col = min(center_full, vs_full.shape[1] - 1)
            af_h = haskell_af_within(
                freq,
                vs_full[:, col],
                zeta_full[:, col],
                dz=dz,
                vs_rock=vs2,
                soil_nz=soil_nz,
            )
            r_h = _rel_l2(af_h, af_ops)
            p_h = _pearson(af_h, af_ops)
            buckets["haskell_rel"].append(r_h)
            buckets["haskell_pear"].append(p_h)
            row = {
                "sobol_id": sid,
                "seed": seed,
                "cov": cov,
                "gate": residual_gate_scalar(cov, 0.0),
                "haskell_rel_l2": r_h,
                "haskell_pearson": p_h,
            }

            if model is not None:
                vs, zeta, x_in, dip_rms, nz = _prepare_context(
                    vs_full, zeta_full, vs2, H, dz, dx
                )
                model_freq = np.logspace(-1, 1, config.N_FREQ)
                af_h_m = np.interp(model_freq, freq, af_h).astype(np.float32)
                haskell_grid = np.zeros((config.NX, config.N_FREQ), dtype=np.float32)
                haskell_grid[center_strip] = af_h_m
                x_t = torch.from_numpy(x_in).unsqueeze(0).to(device)
                h_t = torch.from_numpy(haskell_grid).unsqueeze(0).to(device)
                cov_t = torch.tensor([cov], dtype=torch.float32, device=device)
                dip_t = torch.tensor([dip_rms], dtype=torch.float32, device=device)
                pred = model(x_t, h_t, cov_t, dip_t)[0, center_strip].cpu().numpy()
                pred_i = np.interp(freq, model_freq, pred)
                r_d = _rel_l2(pred_i, af_ops)
                p_d = _pearson(pred_i, af_ops)
                buckets["delta_rel"].append(r_d)
                buckets["delta_pear"].append(p_d)
                row["delta_rel_l2"] = r_d
                row["delta_pearson"] = p_d

            ig = grf_lookup.get((sid, seed))
            if ig is not None:
                gp = args.h5_dir / f"run_{ig}.h5"
                if gp.is_file():
                    with h5py.File(gp, "r") as f:
                        af_g = f["transfer_function"]["AF"][:].astype(np.float64)
                        fg = f["transfer_function"]["freq"][:].astype(np.float64)
                    if fg.shape != freq.shape or not np.allclose(fg, freq, atol=1e-12):
                        af_g = np.interp(freq, fg, af_g)
                    r_g = _rel_l2(af_g, af_ops)
                    p_g = _pearson(af_g, af_ops)
                    buckets["gifno_rel"].append(r_g)
                    buckets["gifno_pear"].append(p_g)
                    row["gifno_rel_l2"] = r_g
                    row["gifno_pearson"] = p_g

            rows.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_path = args.out_dir / "rv_rescore_per_seed.csv"
    with per_path.open("w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    summary = {k: _stats(v) for k, v in buckets.items() if v}
    sum_path = args.out_dir / "rv_rescore_summary.json"
    sum_path.write_text(json.dumps(summary, indent=2))

    print("=== RV re-score vs OpenSees-2D (center |TF|) ===")
    for k, s in summary.items():
        print(
            f"  {k}: n={s['n']} mean={s['mean']:.4f} med={s['median']:.4f} "
            f"p10={s['p10']:.4f} p90={s['p90']:.4f}"
        )
    print(f"Wrote {per_path}")
    print(f"Wrote {sum_path}")


if __name__ == "__main__":
    main()
