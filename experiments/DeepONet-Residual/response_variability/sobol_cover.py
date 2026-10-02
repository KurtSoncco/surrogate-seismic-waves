#!/usr/bin/env python3
"""Train vs val+test Sobol occupancy pairplots (IID 6-D, dipping 7-D).

    uv run python experiments/DeepONet-Residual/response_variability/sobol_cover.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

import config  # noqa: E402
from mix_ladder import mix_test_parts, mix_train_parts, mix_val_lookup  # noqa: E402
from ood_signed_cache import cache_dir_for  # noqa: E402
from response_variability.covariates import COVARIATE_LABELS  # noqa: E402
from response_variability.plot_presentation import DOMAIN_SPECS  # noqa: E402
from response_variability.sobol_design import unique_rows  # noqa: E402
from response_variability.style import apply_nature_style, figsize, savefig  # noqa: E402

OUT_DIR = config.RESULTS_DIR / "response_variability" / "sobol_probe"

# Nested n1000 / dipping design axes (pack / meta keys).
IID_COVER_KEYS: tuple[str, ...] = ("vs1", "H", "cov", "rH", "aHV", "vs2")
DIPPING_COVER_KEYS: tuple[str, ...] = IID_COVER_KEYS + ("dip_angle_deg",)
META_KEY = {
    "vs1": "Vs1",
    "H": "H",
    "cov": "CoV",
    "rH": "rH",
    "aHV": "aHV",
    "vs2": "Vs2",
}
AXIS_LABELS = {
    "vs1": COVARIATE_LABELS["vs1"],
    "H": COVARIATE_LABELS["H"],
    "cov": COVARIATE_LABELS["cov"],
    "rH": COVARIATE_LABELS["rH"],
    "aHV": COVARIATE_LABELS["aHV"],
    "vs2": COVARIATE_LABELS["vs2"],
    "dip_angle_deg": COVARIATE_LABELS["dip_angle_deg"],
}
TRAIN_COLOR = "#BBBBBB"
HELD_COLOR = {
    "iid": "#0072B2",
    "dipping": "#D55E00",
}


def cover_keys(domain: str) -> tuple[str, ...]:
    if domain == "iid":
        return IID_COVER_KEYS
    if domain == "dipping":
        return DIPPING_COVER_KEYS
    raise ValueError(f"cover keys only for iid/dipping, got {domain!r}")


def dipping_dip_angle_table() -> np.ndarray:
    """``dip_angle_deg`` for local cache index 0..n-1 from the OOD manifest."""
    man = config.ood_dipping_root() / "manifest.csv"
    if not man.is_file():
        return np.array([], dtype=float)
    by_index: dict[int, float] = {}
    max_i = -1
    with open(man, newline="") as f:
        for row in csv.DictReader(f):
            i = int(row["index"])
            by_index[i] = float(row["dip_angle_deg"])
            max_i = max(max_i, i)
    if max_i < 0:
        return np.array([], dtype=float)
    table = np.full(max_i + 1, np.nan, dtype=float)
    for i, ang in by_index.items():
        table[i] = ang
    return table


def _meta_matrix(
    cache_dir: Path,
    idx: np.ndarray,
    keys: tuple[str, ...],
    *,
    dip_table: np.ndarray | None = None,
) -> np.ndarray:
    meta = dict(np.load(Path(cache_dir) / "meta.npz", allow_pickle=True))
    loc = np.asarray(idx, dtype=int)
    cols = []
    for key in keys:
        if key == "dip_angle_deg":
            if dip_table is None or dip_table.size == 0:
                cols.append(np.full(len(loc), np.nan))
            else:
                ang = np.full(len(loc), np.nan)
                ok = (loc >= 0) & (loc < dip_table.size)
                ang[ok] = dip_table[loc[ok]]
                cols.append(ang)
            continue
        src = META_KEY[key]
        cols.append(np.asarray(meta[src], dtype=float)[loc])
    return np.column_stack(cols)


def load_split_cloud(domain: str, split: str) -> dict[str, np.ndarray]:
    keys = cover_keys(domain)
    mix_key = DOMAIN_SPECS[domain]["mix_key"]
    dip_table = dipping_dip_angle_table() if domain == "dipping" else None
    if split == "train":
        if domain == "iid":
            cache = config.CACHE_DIR / "n1000_seed42"
            from mix_ladder import iid_n1000_split

            idx = np.asarray(iid_n1000_split()["train"], dtype=int)
        else:
            cache = cache_dir_for("ood_dipping")
            parts = {name: (c, i) for name, c, i in mix_train_parts("M700")}
            cache, idx = parts[mix_key]
            idx = np.asarray(idx, dtype=int)
    elif split == "val":
        cache, idx = mix_val_lookup()[mix_key]
        idx = np.asarray(idx, dtype=int)
    elif split == "test":
        cache, idx = mix_test_parts()[mix_key]
        idx = np.asarray(idx, dtype=int)
    else:
        raise ValueError(split)
    x = _meta_matrix(cache, idx, keys, dip_table=dip_table)
    return {
        "x": x,
        "idx": idx,
        "keys": np.array(keys),
        "n_files": np.array([len(idx)]),
        "n_unique": np.array([len(unique_rows(x))]),
    }


def occupancy_row(domain: str, split: str, cloud: dict[str, np.ndarray]) -> dict:
    return {
        "domain": domain,
        "split": split,
        "n_files": int(np.asarray(cloud["n_files"]).ravel()[0]),
        "n_unique": int(np.asarray(cloud["n_unique"]).ravel()[0]),
        "n_axes": int(len(cloud["keys"])),
    }


def unique_overlap(train_x: np.ndarray, held_x: np.ndarray) -> dict[str, int]:
    t = unique_rows(train_x)
    h = unique_rows(held_x)
    if t.size == 0 or h.size == 0:
        return {"n_train": len(t), "n_held": len(h), "n_overlap": 0, "n_held_only": len(h)}
    t_set = {tuple(np.round(row, 8)) for row in t}
    h_set = {tuple(np.round(row, 8)) for row in h}
    overlap = t_set & h_set
    return {
        "n_train": len(t_set),
        "n_held": len(h_set),
        "n_overlap": len(overlap),
        "n_held_only": len(h_set - t_set),
    }


def plot_cover_pairplot(
    train_x: np.ndarray,
    held_x: np.ndarray,
    keys: tuple[str, ...],
    path: Path,
    *,
    domain: str,
    n_train: int,
    n_held: int,
    n_unique_train: int,
    n_unique_held: int,
) -> Path:
    import matplotlib.pyplot as plt

    apply_nature_style()
    d = len(keys)
    held_c = HELD_COLOR[domain]
    fig, axes = plt.subplots(
        d,
        d,
        figsize=figsize("double", height_mm=28.0 * d + 18.0),
        squeeze=False,
    )
    for i, yi in enumerate(keys):
        for j, xj in enumerate(keys):
            ax = axes[i, j]
            if i == j:
                ax.hist(
                    train_x[:, j],
                    bins=18,
                    color=TRAIN_COLOR,
                    density=True,
                    histtype="stepfilled",
                    alpha=0.85,
                )
                ax.hist(
                    held_x[:, j],
                    bins=18,
                    color=held_c,
                    density=True,
                    histtype="step",
                    linewidth=1.1,
                )
            else:
                ax.scatter(
                    train_x[:, j],
                    train_x[:, i],
                    s=6,
                    c=TRAIN_COLOR,
                    alpha=0.55,
                    linewidths=0,
                    zorder=1,
                )
                ax.scatter(
                    held_x[:, j],
                    held_x[:, i],
                    s=9,
                    c=held_c,
                    alpha=0.8,
                    linewidths=0,
                    zorder=2,
                )
            if i == d - 1:
                ax.set_xlabel(AXIS_LABELS[xj], fontsize=6)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(AXIS_LABELS[yi], fontsize=6)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=5.5)
    title = DOMAIN_SPECS[domain]["title"]
    fig.suptitle(
        rf"{title}: train $n$={n_train} ({n_unique_train} unique) vs "
        rf"val+test $n$={n_held} ({n_unique_held} unique)",
        fontsize=8,
        y=1.01,
    )
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig.legend(
        handles=[
            Patch(facecolor=TRAIN_COLOR, edgecolor="none", label="train"),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=held_c,
                markersize=5,
                label="val+test",
            ),
        ],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
        fontsize=7,
    )
    fig.tight_layout(h_pad=0.15, w_pad=0.15, rect=(0, 0.04, 1, 0.98))
    return savefig(fig, path)


def run(*, out_dir: Path = OUT_DIR) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    rows: list[dict] = []
    overlap_rows: list[dict] = []
    for domain in ("iid", "dipping"):
        keys = cover_keys(domain)
        train = load_split_cloud(domain, "train")
        val = load_split_cloud(domain, "val")
        test = load_split_cloud(domain, "test")
        held_x = np.vstack([val["x"], test["x"]])
        held_n = int(val["x"].shape[0] + test["x"].shape[0])
        held_u = len(unique_rows(held_x))
        for split, cloud in (("train", train), ("val", val), ("test", test)):
            rows.append(occupancy_row(domain, split, cloud))
        rows.append(
            {
                "domain": domain,
                "split": "val+test",
                "n_files": held_n,
                "n_unique": held_u,
                "n_axes": len(keys),
            }
        )
        ov = unique_overlap(train["x"], held_x)
        ov["domain"] = domain
        overlap_rows.append(ov)
        fname = "pairplot_iid_6d.png" if domain == "iid" else "pairplot_dipping_7d.png"
        paths[domain] = plot_cover_pairplot(
            train["x"],
            held_x,
            keys,
            out_dir / fname,
            domain=domain,
            n_train=int(train["n_files"].ravel()[0]),
            n_held=held_n,
            n_unique_train=int(train["n_unique"].ravel()[0]),
            n_unique_held=held_u,
        )
        print(f"Wrote {paths[domain]}", flush=True)
    occ_path = out_dir / "train_val_test_occupancy.csv"
    with open(occ_path, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["domain", "split", "n_files", "n_unique", "n_axes"]
        )
        w.writeheader()
        w.writerows(rows)
    ov_path = out_dir / "train_heldout_id_overlap.csv"
    with open(ov_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["domain", "n_train", "n_held", "n_overlap", "n_held_only"],
        )
        w.writeheader()
        w.writerows(overlap_rows)
    summary = {"occupancy": rows, "overlap": overlap_rows}
    (out_dir / "train_heldout_cover.json").write_text(json.dumps(summary, indent=2))
    paths["occupancy"] = occ_path
    paths["overlap"] = ov_path
    return paths


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args()
    run(out_dir=args.out_dir)


if __name__ == "__main__":
    main()
