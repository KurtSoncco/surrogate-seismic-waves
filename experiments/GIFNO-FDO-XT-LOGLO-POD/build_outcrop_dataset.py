#!/usr/bin/env python3
"""Pack relabeled campaigns into a GIFNO_TF_DIR the training scripts read as-is.

Each ``--source NAME=DIR[:N]`` points at a directory written by
``experiments/case7a/relabel_outcrop.py`` (``manifest.csv`` plus the
``*_outcrop_*`` arrays). Only rows whose H5 was relabeled are kept; ``:N``
keeps a random N of them. ``--holdout-frac`` sets aside a fraction of every
source for evaluation.

``--target outcrop``  tf_per_sample = surface / (c * input)
``--target residual`` tf_per_sample = (surface / (c * input)) / TF_1D, so a
log-POD head learns the multifidelity log residual and the prediction is
TF_1D * exp(head). ``tf1d_per_sample.npy`` is always written for that.

    python build_outcrop_dataset.py --target residual \
        --source iid=../../data/gifno_screen/transfer_function \
        --out ../../data/gifno_screen/outcrop_tf/residual_iid
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

TF_FLOOR = 1e-6


def _parse_source(spec: str) -> tuple[str, Path, int | None]:
    name, _, rest = spec.partition("=")
    if not rest:
        raise ValueError(f"--source must be NAME=DIR[:N], got {spec!r}")
    path, n = rest, None
    head, _, tail = rest.rpartition(":")
    if head and tail.isdigit():
        path, n = head, int(tail)
    return name, Path(path).expanduser().resolve(), n


def _h5_dir(src: Path) -> Path:
    return src / "h5" if (src / "h5").is_dir() else src.parent / "h5"


def _h5_name(row: dict) -> str:
    idx = row.get("run_index") or row.get("index") or row.get("sample_idx")
    return f"run_{int(idx)}.h5"


def _resolve_h5(stored: str | None, h5_dir: Path, src: Path, row: dict) -> str:
    """Keep absolute paths when they exist; otherwise match by filename on this host."""
    if stored and Path(stored).is_file():
        return stored
    name = Path(stored).name if stored else _h5_name(row)
    for folder in (h5_dir, src / "h5", src.parent / "h5"):
        cand = folder / name
        if cand.is_file():
            return str(cand)
    return str(h5_dir / name)


def load_source(name: str, src: Path, n_keep: int | None, rng: np.random.Generator) -> dict:
    rows = list(csv.DictReader(open(src / "manifest.csv")))
    valid = np.load(src / "outcrop_valid.npy")
    if len(valid) != len(rows):
        raise RuntimeError(f"{name}: outcrop_valid has {len(valid)} rows, manifest {len(rows)}")
    keep = np.flatnonzero(valid)
    if n_keep is not None and n_keep < keep.size:
        keep = np.sort(rng.choice(keep, size=n_keep, replace=False))
    tf = np.load(src / "tf_outcrop_per_sample.npy", mmap_mode="r")[keep]
    tf1d = np.load(src / "tf1d_outcrop_per_sample.npy", mmap_mode="r")[keep]
    freq = np.load(src / "outcrop_freq.npy")
    h5_dir = _h5_dir(src)
    out_rows = []
    for i in keep:
        r = dict(rows[i])
        r["h5_path"] = _resolve_h5(r.get("h5_path"), h5_dir, src, r)
        r["campaign"] = name
        r["source_row"] = str(int(i))
        out_rows.append(r)
    return {"rows": out_rows, "tf": np.asarray(tf), "tf1d": np.asarray(tf1d), "freq": freq, "src": src}


def _apply_share(parts: list[dict], spec: str, rng: np.random.Generator) -> None:
    name, _, frac_s = spec.partition("=")
    frac = float(frac_s)
    if not 0.0 < frac < 1.0:
        raise ValueError(f"--share {spec}: FRAC must be in (0, 1)")
    mine = [p for p in parts if p["rows"] and p["rows"][0]["campaign"] == name]
    if not mine:
        raise ValueError(f"--share {spec}: no source named {name}")
    n_mine = sum(len(p["rows"]) for p in mine)
    n_other = sum(len(p["rows"]) for p in parts) - n_mine
    want = int(round(frac / (1.0 - frac) * n_other))
    if want >= n_mine:
        print(f"--share {spec}: {name} has {n_mine} rows, {want} wanted; kept all ({n_mine / (n_mine + n_other):.3f})")
        return
    for p in mine:
        k = int(round(want * len(p["rows"]) / n_mine))
        idx = np.sort(rng.choice(len(p["rows"]), size=k, replace=False))
        p["rows"] = [p["rows"][i] for i in idx]
        p["tf"], p["tf1d"] = p["tf"][idx], p["tf1d"][idx]
    print(f"--share {spec}: {name} {n_mine} -> {want} rows")


def _holdout_mask(rows: list[dict], frac: float, rng: np.random.Generator) -> np.ndarray:
    """Random fraction of rows, or of ``group`` values when rows carry one
    (section windows: all windows of a run land on one side). Rows with
    ``split == validation`` are always held out."""
    n = len(rows)
    forced = np.array([r.get("split") == "validation" for r in rows], dtype=bool)
    groups = [r.get("group") or None for r in rows]
    if n and all(groups):
        uniq = sorted(set(g for g, f in zip(groups, forced) if not f))
        chosen = set(rng.permutation(uniq)[: int(round(frac * len(uniq)))].tolist())
        return forced | np.array([g in chosen for g in groups], dtype=bool)
    perm = rng.permutation(n)
    mask = np.zeros(n, dtype=bool)
    mask[perm[: int(round(frac * n))]] = True
    return forced | mask


def _source_recorder_x(src: Path, n_rec: int, fallback: np.ndarray) -> np.ndarray:
    path = src / "recorder_x_idx.npy"
    if path.is_file():
        rec = np.asarray(np.load(path), dtype=np.int64)
        if rec.size != n_rec:
            raise RuntimeError(f"{src}: recorder_x_idx has {rec.size} slots, tf has {n_rec}")
        return rec
    if fallback.size == n_rec:
        return np.asarray(fallback, dtype=np.int64)
    raise RuntimeError(f"{src}: no recorder_x_idx.npy and tf width {n_rec} != fallback {fallback.size}")


def _align_collocation(parts: list[dict], fallback: np.ndarray) -> tuple[np.ndarray, bool]:
    """Put every source on the union of collocation columns. Returns union, mixed."""
    recs = [_source_recorder_x(p["src"], p["tf"].shape[1], fallback) for p in parts]
    union = np.unique(np.concatenate(recs))
    mixed = any(rec.size != union.size or not np.array_equal(rec, union) for rec in recs)
    if not mixed:
        return union, False
    col = {int(x): i for i, x in enumerate(union)}
    for p, rec in zip(parts, recs):
        n, _, f = p["tf"].shape
        tf = np.zeros((n, union.size, f), dtype=np.float32)
        tf1d = np.zeros_like(tf)
        ok = np.zeros((n, union.size), dtype=bool)
        for j, x in enumerate(rec):
            i = col[int(x)]
            tf[:, i] = p["tf"][:, j]
            tf1d[:, i] = p["tf1d"][:, j]
            ok[:, i] = True
        p["tf"], p["tf1d"], p["collocation_valid"] = tf, tf1d, ok
    return union, True


def _write(
    out: Path,
    rows: list[dict],
    tf: np.ndarray,
    tf1d: np.ndarray,
    freq: np.ndarray,
    recorder_x: np.ndarray,
    meta: dict,
    valid: np.ndarray | None = None,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        keys.extend(k for k in r if k not in keys)
    with open(out / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({**r, "sample_idx": str(i)} if "sample_idx" in keys else r)
    np.save(out / "tf_per_sample.npy", tf.astype(np.float32))
    np.save(out / "tf1d_per_sample.npy", tf1d.astype(np.float32))
    np.save(out / "freq.npy", freq.astype(np.float32))
    np.save(out / "recorder_x_idx.npy", recorder_x)
    if valid is not None:
        np.save(out / "collocation_valid.npy", np.asarray(valid, dtype=bool))
    (out / "dataset.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", action="append", required=True, help="NAME=DIR[:N]")
    ap.add_argument("--target", choices=("outcrop", "residual"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--holdout-frac", type=float, default=0.0)
    ap.add_argument(
        "--share",
        action="append",
        default=[],
        help="NAME=FRAC: subsample source NAME to FRAC of all rows (e.g. iid=0.34); only shrinks",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--recorder-x",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "gifno_screen" / "transfer_function" / "recorder_x_idx.npy",
    )
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    parts = [load_source(*_parse_source(s), rng) for s in args.source]
    for spec in args.share:
        _apply_share(parts, spec, rng)
    freq = parts[0]["freq"]
    for p in parts[1:]:
        if not np.allclose(p["freq"], freq, rtol=1e-6):
            raise RuntimeError(f"frequency grid of {p['src']} differs")

    fallback_rec = np.load(args.recorder_x)
    recorder_x, mixed = _align_collocation(parts, fallback_rec)

    train = {"rows": [], "tf": [], "tf1d": [], "valid": []}
    hold = {"rows": [], "tf": [], "tf1d": [], "valid": []}
    counts = {}
    for p in parts:
        n = len(p["rows"])
        hold_mask = _holdout_mask(p["rows"], args.holdout_frac, rng)
        n_hold = int(hold_mask.sum())
        valid = p.get("collocation_valid")
        for dst, idx in ((hold, np.flatnonzero(hold_mask)), (train, np.flatnonzero(~hold_mask))):
            dst["rows"].extend(p["rows"][i] for i in idx)
            dst["tf"].append(p["tf"][idx])
            dst["tf1d"].append(p["tf1d"][idx])
            if valid is not None:
                dst["valid"].append(valid[idx])
        counts[p["rows"][0]["campaign"] if p["rows"] else str(p["src"])] = {"train": n - n_hold, "holdout": n_hold}

    for split, d, out in (("train", train, args.out), ("holdout", hold, args.out / "holdout")):
        if not d["rows"]:
            continue
        tf = np.concatenate(d["tf"])
        tf1d = np.concatenate(d["tf1d"])
        valid = np.concatenate(d["valid"]) if d["valid"] else None
        if not (np.isfinite(tf).all() and np.isfinite(tf1d).all()):
            raise RuntimeError(f"{split}: non-finite TF values")
        label = tf / np.maximum(tf1d, TF_FLOOR) if args.target == "residual" else tf
        meta = {
            "target": args.target,
            "split": split,
            "n": len(d["rows"]),
            "sources": [str(p["src"]) for p in parts],
            "counts": counts,
            "seed": args.seed,
            "n_collocation": int(recorder_x.size),
            "mixed_collocation": mixed,
        }
        _write(out, d["rows"], label, tf1d, freq, recorder_x, meta, valid=valid)
        print(f"{split}: {len(d['rows'])} rows -> {out}")


if __name__ == "__main__":
    main()
