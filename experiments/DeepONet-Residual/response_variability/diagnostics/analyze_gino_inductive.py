#!/usr/bin/env python3
"""Wave 1 instruments. Pack DCT runs without a ckpt; collapse/orbits/probes need fields.

uv run python experiments/DeepONet-Residual/response_variability/diagnostics/analyze_gino_inductive.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_EXP = Path(__file__).resolve().parents[2]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

import config  # noqa: E402
from response_variability.gino_bias import leftover, to_jsonable  # noqa: E402
from response_variability.gino_inductive import (  # noqa: E402
    extract_iid_branch_latents,
    invariance_orbit_check,
    mean_r_on_train,
    pack_dct_from_leftover,
    prior_collapse_curve,
    represent_vs_use,
)
from response_variability.plots.plot_presentation import load_pack, pack_path  # noqa: E402

PACK_DIR = config.RESULTS_DIR / "presentation"
OUT_DIR = config.RESULTS_DIR / "response_variability" / "gino_bias"


def _maybe_ckpt() -> Path | None:
    p = config.DEFAULT_CHECKPOINT
    return p if p.is_file() else None


def main() -> None:
    p = argparse.ArgumentParser(description="Wave 1 GINO inductive instruments")
    p.add_argument("--pack-dir", type=Path, default=PACK_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--ckpt", type=Path, default=None)
    args = p.parse_args()
    ckpt = args.ckpt if args.ckpt is not None else _maybe_ckpt()
    out: dict[str, Any] = {
        "ckpt": str(ckpt) if ckpt is not None else None,
        "ckpt_present": ckpt is not None,
        "loglo_contrast": False,
        "claim_downgrade_if_no_loglo": (
            "off-support the model reverts to its training mean, which in this "
            "formulation is near-1D"
        ),
    }
    if pack_path(args.pack_dir, "iid").is_file():
        pack = load_pack(pack_path(args.pack_dir, "iid"))
        r_hat = leftover(pack["tf_gino"], pack["tf_haskell_nominal"])
        r = leftover(pack["tf_opensees"], pack["tf_haskell_nominal"])
        out["iid_mean_R"] = mean_r_on_train(r)
        dct = pack_dct_from_leftover(r_hat, r)
        dct.pop("ratio", None)
        out["iid_dct"] = dct
        vs1 = np.asarray(pack["vs1"], dtype=float)
        out["invariance_groups"] = invariance_orbit_check(vs1, pack["vs2"], pack["H"])
        # Label-only proxy collapse: leftover magnitude vs 4D distance to centroid.
        x = np.column_stack([pack["vs1"], pack["H"], pack["cov"], pack["vs2"]]).astype(
            float
        )
        z = (x - x.mean(0)) / np.clip(x.std(0), 1e-8, None)
        dist = np.linalg.norm(z, axis=1)
        r_norm = np.linalg.norm(r.reshape(r.shape[0], -1), axis=1)
        r_norm = r_norm / np.clip(
            np.linalg.norm(
                np.asarray(pack["tf_haskell_nominal"]).reshape(r.shape[0], -1), axis=1
            ),
            1e-12,
            None,
        )
        out["prior_collapse_proxy"] = prior_collapse_curve(
            dist, r_norm, hull=float(np.median(dist))
        )
        rng = np.random.default_rng(0)
        # Pack-level represent-vs-use on 4D scalars as a *negative control* for the
        # probe plumbing (not a latent probe). Real GNO latents need the ckpt.
        latent = np.column_stack([z, rng.normal(size=(z.shape[0], 8))])
        f0 = vs1 / np.clip(4.0 * np.asarray(pack["H"], dtype=float), 1e-12, None)
        err = np.linalg.norm(
            (pack["tf_gino"] - pack["tf_opensees"]).reshape(r.shape[0], -1), axis=1
        )
        probe = represent_vs_use(latent, f0, err, seed=0)
        probe.pop("probe_pred_te", None)
        out["scalar_probe_control"] = probe
    else:
        out["skipped_packs"] = True
    if ckpt is None:
        out["wave1_model_probes"] = "skipped_no_checkpoint"
        out["prior_collapse_synthetic_fields"] = "skipped_no_checkpoint"
        out["invariance_orbits_on_fields"] = "skipped_no_checkpoint"
    else:
        cache = config.CACHE_DIR / "n1000_seed42"
        fields_ok = (cache / "fields.npy").is_file() and (
            cache / "r_nom_signed.npy"
        ).is_file()
        if not fields_ok:
            out["wave1_model_probes"] = "skipped_no_iid_cache"
        else:
            from mix_ladder import iid_n1000_split  # noqa: E402
            from response_variability.gino_bias import per_sample_rel_l2  # noqa: E402

            test_idx = iid_n1000_split()["test"]
            lat = extract_iid_branch_latents(
                ckpt_path=ckpt,
                cache_dir=cache,
                test_idx=test_idx,
                n_freq=config.N_FREQ_EVAL,
            )
            pack = load_pack(pack_path(args.pack_dir, "iid"))
            vs1 = np.asarray(pack["vs1"], dtype=float)
            H = np.asarray(pack["H"], dtype=float)
            f0 = vs1 / np.clip(4.0 * H, 1e-12, None)
            cov = np.asarray(pack["cov"], dtype=float)
            err = per_sample_rel_l2(pack["tf_gino"], pack["tf_opensees"])
            n = min(lat["branch"].shape[0], f0.shape[0], err.shape[0])
            f0_probe = represent_vs_use(lat["branch"][:n], f0[:n], err[:n], seed=0)
            f0_probe.pop("probe_pred_te", None)
            cov_probe = represent_vs_use(lat["branch"][:n], cov[:n], err[:n], seed=1)
            cov_probe.pop("probe_pred_te", None)
            nv = lat["node_spatial_var"][:n]
            cv = cov[:n]
            finite = np.isfinite(nv) & np.isfinite(cv)
            if finite.sum() >= 8:
                ra = np.argsort(np.argsort(cv[finite]))
                rb = np.argsort(np.argsort(nv[finite]))
                node_sp = float(np.corrcoef(ra, rb)[0, 1])
            else:
                node_sp = float("nan")
            out["gno_latent_probes"] = {
                "f0": f0_probe,
                "CoV": cov_probe,
                "node_var_vs_cov_spearman": node_sp,
                "n": float(n),
                "latent_dim": float(lat["branch"].shape[1]),
            }
            out["prior_collapse_synthetic_fields"] = (
                "not run: needs RF synthesis past the hull; pack proxy is in "
                "prior_collapse_proxy. LOGLO contrast not stood up — claim downgraded."
            )
            out["invariance_orbits_on_fields"] = (
                "group maps checked in invariance_groups; full c_v/c_l field "
                "re-encode orbits need a generator that scales all lengths"
            )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "wave1.json"
    path.write_text(json.dumps(to_jsonable(out), indent=2))
    print(f"Wrote {path}  ckpt={ckpt is not None}", flush=True)


if __name__ == "__main__":
    main()
