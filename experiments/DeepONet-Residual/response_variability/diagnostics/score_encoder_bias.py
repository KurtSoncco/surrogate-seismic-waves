#!/usr/bin/env python3
"""D0: rebuild nested presentation packs per leftover ckpt and score bias instruments.

Does not overwrite ship packs in results/presentation/{iid,dipping,three_layer}_pack.npz.
Writes results/presentation/<run>/ and results/response_variability/gino_bias/<run>/.

    uv run python experiments/DeepONet-Residual/response_variability/diagnostics/score_encoder_bias.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

_EXP = Path(__file__).resolve().parents[2]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

import config  # noqa: E402
from response_variability.plots.plot_presentation import (  # noqa: E402
    DOMAIN_SPECS,
    build_domain_pack,
    pack_path,
)

SHIP_PACK_DIR = config.RESULTS_DIR / "presentation"
CKPT_DIR = config.CHECKPOINT_DIR
RUNS: tuple[tuple[str, str, bool], ...] = (
    ("M7680_gino_rebal_ft", "M7680_gino_rebal_ft.pt", False),
    ("gino_waveA_noop", "gino_waveA_noop.pt", False),
    ("gino_gno_unfreeze_probe", "gino_gno_unfreeze_probe.pt", False),
    ("gino_waveC_logR", "gino_waveC_logR.pt", False),
    ("gino_waveC_band2", "gino_waveC_band2.pt", False),
)
CONTROL = "gino_waveA_noop"


def _q4_mean(blob: dict[str, Any], axis: str) -> float:
    rows = ((blob.get("quartile_rel_l2") or {}).get(axis)) or []
    for row in rows:
        if int(row.get("quartile", 0)) == 4:
            return float(row["mean"])
    return float("nan")


def _copy_ship_packs(dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for domain in DOMAIN_SPECS:
        src = pack_path(SHIP_PACK_DIR, domain)
        if not src.is_file():
            raise FileNotFoundError(src)
        target = pack_path(dst, domain)
        if not target.is_file():
            shutil.copy2(src, target)


def _packs_ready(pack_dir: Path) -> bool:
    return all(pack_path(pack_dir, d).is_file() for d in DOMAIN_SPECS)


def _run_py(script: str, *args: str) -> None:
    cmd = [sys.executable, "-u", str(_EXP / "response_variability" / script), *args]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(_EXP))


def score_run(
    name: str,
    ckpt: Path,
    *,
    reuse_ship_packs: bool,
    n_pretell: int,
    batch_size: int,
    skip_predict: bool,
) -> None:
    pack_dir = config.RESULTS_DIR / "presentation" / name
    bias_dir = config.RESULTS_DIR / "response_variability" / "gino_bias" / name
    pack_dir.mkdir(parents=True, exist_ok=True)
    if reuse_ship_packs:
        _copy_ship_packs(pack_dir)
    elif not (skip_predict and _packs_ready(pack_dir)):
        for domain in DOMAIN_SPECS:
            out = pack_path(pack_dir, domain)
            if skip_predict and out.is_file():
                continue
            if out.is_file() and not skip_predict:
                print(f"SKIP pack {out}", flush=True)
                continue
            print(f"=== pack {name}/{domain} ===", flush=True)
            build_domain_pack(
                domain,
                ckpt_path=ckpt,
                out_dir=pack_dir,
                batch_size=batch_size,
                n_pretell=n_pretell,
                skip_predict=False,
            )
    if not (bias_dir / "summary.json").is_file():
        _run_py(
            "analyze_gino_bias.py",
            "--pack-dir",
            str(pack_dir),
            "--out-dir",
            str(bias_dir),
        )
    else:
        print(f"SKIP bias {bias_dir / 'summary.json'}", flush=True)
    if not (bias_dir / "wave1.json").is_file():
        _run_py(
            "analyze_gino_inductive.py",
            "--ckpt",
            str(ckpt),
            "--pack-dir",
            str(pack_dir),
            "--out-dir",
            str(bias_dir),
        )
    else:
        print(f"SKIP wave1 {bias_dir / 'wave1.json'}", flush=True)


def _row(name: str) -> dict[str, Any]:
    bias_dir = config.RESULTS_DIR / "response_variability" / "gino_bias" / name
    summary = json.loads((bias_dir / "summary.json").read_text())
    wave1_path = bias_dir / "wave1.json"
    wave1 = json.loads(wave1_path.read_text()) if wave1_path.is_file() else {}
    iid = (summary.get("domains") or {}).get("iid") or {}
    dip = (summary.get("domains") or {}).get("dipping") or {}
    tl = (summary.get("domains") or {}).get("three_layer") or {}
    floor = iid.get("scalar_floor") or {}
    probes = wave1.get("gno_latent_probes") or {}
    f0 = probes.get("f0") or {}
    cov = probes.get("CoV") or {}
    return {
        "name": name,
        "collapse_ratio": floor.get("collapse_ratio"),
        "r2_scalar": floor.get("r2_scalar_exact"),
        "b": (iid.get("slope") or {}).get("b"),
        "iid_rel_l2": iid.get("rel_l2_mean"),
        "dipping_rel_l2": dip.get("rel_l2_mean"),
        "three_layer_rel_l2": tl.get("rel_l2_mean"),
        "cov_q4_dlnA": _q4_mean(iid, "CoV_delta_lnA"),
        "cov_q4_rel_l2": _q4_mean(iid, "CoV"),
        "spatial_sigma_ratio": (iid.get("spatial") or {}).get("ratio_mean"),
        "f0_probe": {
            "represented_but_unused": f0.get("represented_but_unused"),
            "represented_and_used": f0.get("represented_and_used"),
            "r2_mlp": f0.get("r2_mlp"),
        },
        "cov_probe": {
            "represented_and_used": cov.get("represented_and_used"),
            "represented_but_unused": cov.get("represented_but_unused"),
            "r2_mlp": cov.get("r2_mlp"),
        },
    }


def write_table(names: list[str], out: Path) -> dict[str, Any]:
    rows = [_row(n) for n in names if (config.RESULTS_DIR / "response_variability" / "gino_bias" / n / "summary.json").is_file()]
    control = next((r for r in rows if r["name"] == CONTROL), None)
    did: list[dict[str, Any]] = []
    for r in rows:
        if control is None or r["name"] == CONTROL:
            continue
        def d(key: str) -> float | None:
            a, b = r.get(key), control.get(key)
            if a is None or b is None:
                return None
            return float(a) - float(b)

        did.append(
            {
                "name": r["name"],
                "collapse_ratio": d("collapse_ratio"),
                "cov_q4_dlnA": d("cov_q4_dlnA"),
                "iid_rel_l2": d("iid_rel_l2"),
                "three_layer_rel_l2": d("three_layer_rel_l2"),
            }
        )
    unfreeze = next((r for r in rows if r["name"] == "gino_gno_unfreeze_probe"), None)
    collapse = float("nan") if unfreeze is None else float(unfreeze["collapse_ratio"] or float("nan"))
    import math

    if unfreeze is None or not math.isfinite(collapse) or collapse < 0.7:
        pick = "col_enc_depth_tokens"
    else:
        pick = "unfrozen_modes_up"
    report = {
        "control": CONTROL,
        "rows": rows,
        "did_vs_noop": did,
        "unfreeze_collapse": collapse,
        "d1_pick": pick,
        "rule": "If unfreeze collapse_ratio < 0.7: stop depth pooling (default). Else unfrozen modes-up from unfreeze ckpt.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f"Wrote {out}", flush=True)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt-dir", type=Path, default=CKPT_DIR)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--n-pretell", type=int, default=0, help="0 skips Pretell (bias table does not need it).")
    p.add_argument("--skip-predict", action="store_true")
    p.add_argument("--table-only", action="store_true")
    p.add_argument(
        "--gate-d1",
        action="store_true",
        help="Exit 0 if D1 should be col_enc depth tokens; 1 if unfrozen modes-up.",
    )
    args = p.parse_args()
    names: list[str] = []
    if not args.table_only and not args.gate_d1:
        for name, fname, reuse in RUNS:
            ckpt = args.ckpt_dir / fname
            if not ckpt.is_file():
                print(f"MISSING ckpt {ckpt}", flush=True)
                continue
            names.append(name)
            score_run(
                name,
                ckpt,
                reuse_ship_packs=reuse,
                n_pretell=args.n_pretell,
                batch_size=args.batch_size,
                skip_predict=args.skip_predict,
            )
    else:
        names = [n for n, _, _ in RUNS]
    report = write_table(
        names,
        config.RESULTS_DIR / "response_variability" / "gino_bias" / "d0_encoder_table.json",
    )
    if args.gate_d1:
        pick = str(report.get("d1_pick") or "")
        print(f"D1_GATE {pick}", flush=True)
        sys.exit(0 if pick == "col_enc_depth_tokens" else 1)


if __name__ == "__main__":
    main()
