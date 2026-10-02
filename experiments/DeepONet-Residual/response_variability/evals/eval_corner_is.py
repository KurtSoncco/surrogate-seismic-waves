#!/usr/bin/env python3
"""Score shipped GINO + 1-D arms on the importance-sampled corner OpenSees set.

``data/corner_is/h5/run_{0..159}.h5`` is 32 new 6D locations × 5 RF seeds.
Locations with ``held_out=1`` (8 of 32) are never-train; the other 24 are
train-eligible for a future mix. The shipped checkpoint saw none of them.

    uv run python experiments/DeepONet-Residual/response_variability/evals/eval_corner_is.py
    uv run python .../eval_corner_is.py --skip-predict
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_EXP = Path(__file__).resolve().parents[2]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

import config  # noqa: E402

from ood_signed_cache import build_ood_signed_cache  # noqa: E402
from response_variability.covariates import attach_extracted_f0  # noqa: E402
from response_variability.evals.eval_classical import add_classical_1d_arms  # noqa: E402
from response_variability.evals.eval_iid import summarize_methods  # noqa: E402
from response_variability.plots.plot_eval_bias import (  # noqa: E402
    CORNER_PEARSON_PANELS,
    plot_pearson_boxes_corner,
)
from response_variability.plots.plot_presentation import (  # noqa: E402
    N_PRETELL_DEFAULT,
    attach_vs_and_pretell,
    load_domain_arrays,
    load_pack,
    save_pack,
    score_gino,
)

N_CORNER = 160  # exclude sample-50 extra seeds (run_160..169)
OUT_DIR = config.RESULTS_DIR / "response_variability" / "eval_bias"
PACK_PATH = OUT_DIR / "corner_is_pack.npz"
SUMMARY_PATH = OUT_DIR / "corner_is_summary.csv"


def held_out_by_sample_id(locations_csv: Path | None = None) -> dict[int, int]:
    path = Path(locations_csv or (OUT_DIR / "corner_is_locations.csv"))
    loc = pd.read_csv(path)
    return {int(r.sample_id): int(r.held_out) for r in loc.itertuples()}


def attach_corner_split(
    pack: dict[str, np.ndarray], *, locations_csv: Path | None = None
) -> dict[str, np.ndarray]:
    """Tag each cache row with sample_id / held_out from the IS manifest."""
    n = int(pack["tf_opensees"].shape[0])
    all_csv = OUT_DIR / "corner_is_all.csv"
    if not all_csv.is_file():
        raise FileNotFoundError(all_csv)
    man = pd.read_csv(all_csv)
    loc_idx = np.asarray(pack.get("local_idx", np.arange(n)), dtype=int)
    sample_id = np.full(n, -1, dtype=int)
    held = np.zeros(n, dtype=int)
    held_map = held_out_by_sample_id(locations_csv)
    for i, li in enumerate(loc_idx):
        if 0 <= int(li) < len(man):
            sid = int(man.iloc[int(li)]["sample_id"])
            sample_id[i] = sid
            held[i] = int(held_map.get(sid, 0))
    out = dict(pack)
    out["sample_id"] = sample_id
    out["held_out"] = held
    return out


def split_corner_summaries(summary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if "held_out" not in summary.columns:
        raise KeyError("summary needs held_out")
    flag = summary["held_out"].astype(int)
    return {
        "corner_train": summary.loc[flag == 0].copy(),
        "corner_held": summary.loc[flag == 1].copy(),
    }


def _tag_summary(summary: pd.DataFrame, pack: dict[str, np.ndarray]) -> pd.DataFrame:
    out = summary.copy()
    loc = np.asarray(out["local_idx"], dtype=int)
    out["held_out"] = np.asarray(pack["held_out"], dtype=int)[loc]
    out["sample_id"] = np.asarray(pack["sample_id"], dtype=int)[loc]
    return out


def build_corner_pack(
    *,
    ckpt_path: Path,
    batch_size: int,
    n_pretell: int,
    n_hallal_seeds: int,
    skip_predict: bool,
) -> dict[str, np.ndarray]:
    if skip_predict:
        if not PACK_PATH.is_file():
            raise FileNotFoundError(f"--skip-predict needs {PACK_PATH}")
        pack = load_pack(PACK_PATH)
        if "held_out" not in pack:
            pack = attach_corner_split(pack)
        if "f0_calc" not in pack:
            pack = attach_extracted_f0(pack)
        return pack

    cache_dir = build_ood_signed_cache("corner_is")
    n_cache = int(np.load(cache_dir / "tf2d.npy", mmap_mode="r").shape[0])
    n = min(N_CORNER, n_cache)
    idx = np.arange(n, dtype=int)
    pack = load_domain_arrays(cache_dir, idx)
    pack["domain"] = np.array("corner_is")
    pack["split"] = np.full(n, "corner")
    pack["tf_gino"] = score_gino(cache_dir, idx, ckpt_path, batch_size)
    pack = attach_vs_and_pretell(pack, domain="iid", n_pretell=n_pretell)
    pack = add_classical_1d_arms(
        pack, n_hallal_seeds=n_hallal_seeds, skip_if_present=False
    )
    pack = attach_extracted_f0(pack)
    pack = attach_corner_split(pack)
    save_pack(pack, PACK_PATH)
    print(f"Wrote {PACK_PATH}", flush=True)
    return pack


def run(
    *,
    ckpt_path: Path = config.DEFAULT_CHECKPOINT,
    batch_size: int = config.BATCH_SIZE,
    n_pretell: int = N_PRETELL_DEFAULT,
    n_hallal_seeds: int = 40,
    skip_predict: bool = False,
    skip_plot: bool = False,
) -> dict[str, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pack = build_corner_pack(
        ckpt_path=ckpt_path,
        batch_size=batch_size,
        n_pretell=n_pretell,
        n_hallal_seeds=n_hallal_seeds,
        skip_predict=skip_predict,
    )
    summary, peaks = summarize_methods(pack)
    summary = _tag_summary(summary, pack)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    peaks.to_csv(SUMMARY_PATH.with_name("corner_is_summary_peaks.csv"), index=False)
    slices = split_corner_summaries(summary)
    written: dict[str, Path] = {"summary": SUMMARY_PATH}
    if not skip_plot:
        fig = plot_pearson_boxes_corner(slices, OUT_DIR / "method_ranking_pearson_corner.png")
        written["figure"] = fig
        print(f"Wrote {fig}", flush=True)
    n_train = int((pack["held_out"] == 0).sum())
    n_held = int((pack["held_out"] == 1).sum())
    print(f"[corner_is] n={pack['tf_opensees'].shape[0]} train-eligible={n_train} held-out={n_held}")
    for key, _title in CORNER_PEARSON_PANELS:
        print(f"  {key} rows={len(slices[key])}", flush=True)
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=config.DEFAULT_CHECKPOINT)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--n-pretell", type=int, default=N_PRETELL_DEFAULT)
    p.add_argument("--n-hallal-seeds", type=int, default=40)
    p.add_argument("--skip-predict", action="store_true")
    p.add_argument("--skip-plot", action="store_true")
    args = p.parse_args()
    run(
        ckpt_path=args.checkpoint,
        batch_size=args.batch_size,
        n_pretell=args.n_pretell,
        n_hallal_seeds=args.n_hallal_seeds,
        skip_predict=args.skip_predict,
        skip_plot=args.skip_plot,
    )


if __name__ == "__main__":
    main()
