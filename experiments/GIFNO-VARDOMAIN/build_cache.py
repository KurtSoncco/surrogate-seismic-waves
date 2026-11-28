#!/usr/bin/env python3
"""Pack IID runs and dense-labeled sections into the arrays train.py reads.

    python build_cache.py \
        --iid-h5-dir  .../data/h5 --iid-tf-dir .../data/transfer_function \
        --section round0=.../case7_sections/round0/h5:../../data/vardomain/section_labels/round0 \
        --out ../../data/vardomain/cache

OUT/iid/*.npy
    vs, zeta      (N, NZ_IID, 500) float16, the 500 m variable strip. The 1500 m
                  mesh is this strip plus 500 m of repeated edge column each side
                  (exact in the H5), so any window up to 1500 m is rebuilt from it.
                  Rows past n_active repeat the bottom row.
    n_active      (N,) mesh rows
    log_tf        (N, 21, F) float16 log outcrop TF (relabel_outcrop.py labels)
    valid         (N,) bool
    rec_x         (21,) station columns in the strip
    sample_id, run_index, split   (N,)
OUT/sections/<round>__<sample_id>.npz
    vs, zeta (nz, Lx) float32, x_m (n_nodes,), log_tf (n_nodes, F) float32, split
OUT/freq.npy, OUT/meta.json

IID splits are 80/10/10 by Sobol sample_id (all RF seeds of a sample stay
together); sections hold out about 1/8 for test and 1/10 for validation per
round, by run. The H5 ``split`` attribute of sections is ignored (round0 marks
every run train).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

CASE7A_DIR = Path(__file__).resolve().parents[1] / "case7a"
if str(CASE7A_DIR) not in sys.path:
    sys.path.insert(0, str(CASE7A_DIR))

STRIP0, STRIP1 = 500, 1000
NZ_IID = 110
LOG_FLOOR = 1e-6
SPLIT_SEED = 42


# relabel_outcrop.DAMPING_FREQ_SECOND. The stored 1D prior was matched at this
# frequency, including runs whose H5 has no damping_freq_second attribute.
RAYLEIGH_F2_HZ = 10.0


def _read_iid(path: Path, f1_fallback: float | None = None) -> tuple[np.ndarray, np.ndarray, int, int, float, float]:
    import h5py
    import hdf5plugin  # noqa: F401

    with h5py.File(path, "r") as f:
        vs = np.asarray(f["Vs_realization_2D"][:, STRIP0:STRIP1], dtype=np.float32)
        zeta = np.asarray(f["Damping_zeta"][:, STRIP0:STRIP1], dtype=np.float32)
        sid = int(f.attrs.get("sample_id", -1))
        attrs = f["params"].attrs
        if "damping_freq_first" in attrs:
            f1 = float(attrs["damping_freq_first"])
        elif f1_fallback is not None:
            f1 = float(f1_fallback)
        else:
            raise KeyError(f"{path}: no damping_freq_first")
        f2 = float(attrs["damping_freq_second"]) if "damping_freq_second" in attrs else RAYLEIGH_F2_HZ
    return vs, zeta, vs.shape[0], sid, f1, f2


def _pad_rows(a: np.ndarray, nz: int) -> np.ndarray:
    if a.shape[0] >= nz:
        return a[:nz]
    return np.concatenate([a, np.repeat(a[-1:], nz - a.shape[0], axis=0)], axis=0)


def _split_groups(groups: np.ndarray, test_frac: float, val_frac: float, seed: int = SPLIT_SEED) -> np.ndarray:
    uniq = np.unique(groups)
    perm = np.random.default_rng(seed).permutation(uniq)
    n_test = max(1, int(round(test_frac * uniq.size)))
    n_val = max(1, int(round(val_frac * uniq.size)))
    label = {g: "test" for g in perm[:n_test]}
    label.update({g: "val" for g in perm[n_test : n_test + n_val]})
    return np.array([label.get(g, "train") for g in groups])


def build_iid(h5_dir: Path, tf_dir: Path, out: Path, limit: int | None, workers: int) -> np.ndarray:
    rows = list(csv.DictReader(open(tf_dir / "manifest.csv")))
    tf = np.load(tf_dir / "tf_outcrop_per_sample.npy", mmap_mode="r")
    valid = np.load(tf_dir / "outcrop_valid.npy")
    rec_x = np.load(tf_dir / "recorder_x_idx.npy")
    if len(rows) != tf.shape[0]:
        raise ValueError(f"manifest {len(rows)} rows vs TF {tf.shape[0]}")
    idx = np.arange(len(rows)) if limit is None else np.sort(np.random.default_rng(0).choice(len(rows), limit, replace=False))
    run_index = np.array([int(rows[i]["run_index"]) for i in idx])
    paths = [h5_dir / f"run_{r}.h5" for r in run_index]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} IID H5 missing, first {missing[0]}")
    with ThreadPoolExecutor(workers) as pool:
        fields = list(pool.map(_read_iid, paths))
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "vs.npy", np.stack([_pad_rows(v, NZ_IID) for v, _, _, _, _, _ in fields]).astype(np.float16))
    np.save(out / "zeta.npy", np.stack([_pad_rows(z, NZ_IID) for _, z, _, _, _, _ in fields]).astype(np.float16))
    np.save(out / "n_active.npy", np.array([min(n, NZ_IID) for _, _, n, _, _, _ in fields], dtype=np.int32))
    np.save(out / "rayleigh_f1.npy", np.array([f1 for *_, f1, _ in fields], dtype=np.float32))
    np.save(out / "rayleigh_f2.npy", np.array([f2 for *_, f2 in fields], dtype=np.float32))
    sample_id = np.array([s for _, _, _, s, _, _ in fields], dtype=np.int64)
    np.save(out / "sample_id.npy", sample_id)
    np.save(out / "run_index.npy", run_index)
    np.save(out / "log_tf.npy", np.log(np.maximum(np.asarray(tf[idx]), LOG_FLOOR)).astype(np.float16))
    np.save(out / "valid.npy", valid[idx])
    np.save(out / "rec_x.npy", rec_x.astype(np.int64))
    split = _split_groups(sample_id, 0.1, 0.1)
    np.save(out / "split.npy", split)
    print(f"iid: {idx.size} runs, splits {dict(zip(*np.unique(split, return_counts=True)))}", flush=True)
    return np.load(tf_dir / "outcrop_freq.npy")


def split_by_group(groups: np.ndarray, seed: int = SPLIT_SEED) -> np.ndarray:
    """208/24/24 when there are 256 groups. Smaller campaigns use the same 24/256 fraction."""
    uniq = np.unique(groups)
    perm = np.random.default_rng(seed).permutation(uniq)
    n_test = int(round(24 / 256 * uniq.size))
    n_val = int(round(24 / 256 * uniq.size))
    if n_test + n_val >= uniq.size:
        n_test = max(0, uniq.size // 5)
        n_val = max(0, uniq.size // 5)
    label = {int(g): "train" for g in uniq}
    for g in perm[:n_test]:
        label[int(g)] = "test"
    for g in perm[n_test : n_test + n_val]:
        label[int(g)] = "val"
    return np.array([label[int(g)] for g in groups])


def _run_id(row: dict) -> int:
    if row.get("run_index"):
        return int(row["run_index"])
    return int(row["index"])


def _group_keys(rows: list[dict], kind: str) -> np.ndarray:
    if kind == "iid":
        raw = [(r["H_discretized"], r["CoV"], r["f0_effective"]) for r in rows]
    else:
        raw = [r["sobol_id"] for r in rows]
    codes = {key: i for i, key in enumerate(dict.fromkeys(raw))}
    return np.array([codes[key] for key in raw], dtype=np.int64)


def build_fixed_campaign(name: str, h5_dir: Path, tf_dir: Path, manifest: Path, recorder: Path, out: Path, workers: int) -> np.ndarray:
    """One campaign from the fixed-window labels. ``name`` is ``iid`` or an OOD set."""
    rows = list(csv.DictReader(manifest.open()))
    tf = np.load(tf_dir / "tf_outcrop_per_sample.npy", mmap_mode="r")
    tf1d = np.load(tf_dir / "tf1d_outcrop_per_sample.npy", mmap_mode="r")
    valid = np.load(tf_dir / "outcrop_valid.npy")
    if len(rows) != tf.shape[0] or tf1d.shape != tf.shape:
        raise ValueError(f"{name}: manifest {len(rows)} vs tf {tf.shape} vs tf1d {tf1d.shape}")
    if (out / "vs.npy").is_file() and (out / "log_tf.npy").is_file() and (out / "split.npy").is_file():
        n = int(np.load(out / "n_active.npy").shape[0])
        np.save(out / "rayleigh_f2.npy", np.full(n, RAYLEIGH_F2_HZ, np.float32))
        print(f"{name}: reusing {n} cached runs; Rayleigh f2 = {RAYLEIGH_F2_HZ:g} Hz", flush=True)
        return np.load(tf_dir / "outcrop_freq.npy")
    paths = [h5_dir / f"run_{_run_id(r)}.h5" for r in rows]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{name}: {len(missing)} H5 missing, first {missing[0]}")
    fallback = [r.get("damping_freq_first") or None for r in rows]
    with ThreadPoolExecutor(workers) as pool:
        fields = list(pool.map(_read_iid, paths, fallback))
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "vs.npy", np.stack([_pad_rows(v, NZ_IID) for v, _, _, _, _, _ in fields]).astype(np.float16))
    np.save(out / "zeta.npy", np.stack([_pad_rows(z, NZ_IID) for _, z, _, _, _, _ in fields]).astype(np.float16))
    np.save(out / "n_active.npy", np.array([min(n, NZ_IID) for _, _, n, _, _, _ in fields], dtype=np.int32))
    np.save(out / "rayleigh_f1.npy", np.array([f1 for *_, f1, _ in fields], dtype=np.float32))
    np.save(out / "rayleigh_f2.npy", np.array([f2 for *_, f2 in fields], dtype=np.float32))
    group = _group_keys(rows, "iid" if name == "iid" else "sobol")
    np.save(out / "group.npy", group)
    np.save(out / "sample_id.npy", group)
    np.save(out / "run_index.npy", np.array([_run_id(r) for r in rows], dtype=np.int64))
    np.save(out / "log_tf.npy", np.log(np.maximum(np.asarray(tf), LOG_FLOOR)).astype(np.float16))
    np.save(out / "log_tf1d.npy", np.log(np.maximum(np.asarray(tf1d), LOG_FLOOR)).astype(np.float16))
    np.save(out / "valid.npy", np.asarray(valid))
    np.save(out / "rec_x.npy", np.load(recorder).astype(np.int64))
    split = split_by_group(group)
    np.save(out / "split.npy", split)
    counts = dict(zip(*np.unique(split, return_counts=True)))
    n_groups = {s: len({g for g, lab in zip(group, split) if lab == s}) for s in ("train", "val", "test")}
    print(f"{name}: {len(rows)} runs, rows {counts}, groups {n_groups}", flush=True)
    return np.load(tf_dir / "outcrop_freq.npy")


def manifold_holdout(rows: list[dict], seed: int = SPLIT_SEED) -> dict[str, str]:
    """Blind stays blind. About 12% of the other group_ids in each stratum x width become val."""
    cells: dict[tuple[str, str], list[str]] = {}
    blind = set()
    for r in rows:
        if r.get("status") not in (None, "", "complete"):
            continue
        if r.get("split") == "blind" or r.get("stratum") == "blind":
            blind.add(r["sample_id"])
            continue
        cells.setdefault((r["stratum"], r["lx_m"]), []).append(r["group_id"])
    rng = np.random.default_rng(seed)
    val_groups: set[str] = set()
    for groups in cells.values():
        uniq = list(dict.fromkeys(groups))
        if len(uniq) < 4:
            continue
        n_val = max(1, int(round(0.12 * len(uniq))))
        picked = np.atleast_1d(rng.choice(np.array(uniq), size=n_val, replace=False))
        val_groups.update(str(g) for g in picked)
    out = {}
    for r in rows:
        sid = r["sample_id"]
        if sid in blind or r.get("stratum") == "blind":
            out[sid] = "blind"
        elif r["group_id"] in val_groups:
            out[sid] = "val"
        else:
            out[sid] = "train"
    return out


def build_manifold(h5_dir: Path, label_dir: Path, manifest: Path, out: Path) -> np.ndarray:
    import h5py
    import hdf5plugin  # noqa: F401

    from applied_damping import applied_zeta

    rows = [r for r in csv.DictReader(manifest.open()) if r.get("status") == "complete"]
    split_of = manifold_holdout(rows)
    by_id = {r["sample_id"]: r for r in rows}
    labels = sorted(label_dir.glob("*.npz"))
    if not labels:
        raise FileNotFoundError(f"no manifold labels in {label_dir}")
    out.mkdir(parents=True, exist_ok=True)
    freq = None
    counts: dict[str, int] = {}
    for p in labels:
        lab = np.load(p)
        freq = lab["freq"]
        sid = p.stem
        row = by_id.get(sid)
        if row is None:
            raise KeyError(f"{sid} is not a complete manifest row")
        pad = int(round(float(lab["pad_m"])))
        lx = int(round(float(lab["lx_m"])))
        with h5py.File(h5_dir / f"{sid}.h5", "r") as f:
            vs = np.asarray(f["Vs_realization_2D"][:, pad : pad + lx], dtype=np.float32)
            zeta = applied_zeta(f)[:, pad : pad + lx].astype(np.float32)
            f1 = float(f["params"].attrs["damping_freq_first"])
            f2 = float(f["params"].attrs["damping_freq_second"])
        if vs.shape[1] != lx:
            raise ValueError(f"{sid}: cropped Vs width {vs.shape[1]} != interior {lx}")
        sp = split_of[sid]
        counts[sp] = counts.get(sp, 0) + 1
        np.savez(
            out / f"manifold__{sid}.npz",
            vs=vs, zeta=zeta, x_m=lab["x_m"],
            log_tf=np.log(np.maximum(lab["tf"], LOG_FLOOR)).astype(np.float32),
            log_tf1d=np.log(np.maximum(lab["tf1d"], LOG_FLOOR)).astype(np.float32),
            split=np.array(sp), f1_hz=np.float32(f1), f2_hz=np.float32(f2),
            stratum=np.array(row["stratum"]), group_id=np.array(row["group_id"]),
            damping=np.array(row["damping_family"]),
        )
    print(f"manifold: {len(labels)} interiors, splits {counts}", flush=True)
    return freq


def build_sections(name: str, h5_dir: Path, label_dir: Path, out: Path) -> np.ndarray:
    import h5py
    import hdf5plugin  # noqa: F401

    labels = sorted(label_dir.glob("*.npz"))
    if not labels:
        raise FileNotFoundError(f"no dense labels in {label_dir}")
    sids = np.array([p.stem for p in labels])
    split = _split_groups(sids, 0.125, 0.1)
    out.mkdir(parents=True, exist_ok=True)
    freq = None
    for p, sid, sp in zip(labels, sids, split):
        lab = np.load(p)
        freq = lab["freq"]
        with h5py.File(h5_dir / f"{sid}.h5", "r") as f:
            from applied_damping import applied_zeta

            vs = np.asarray(f["Vs_realization_2D"][:], dtype=np.float32)
            zeta = applied_zeta(f).astype(np.float32)
            dx = float(f["grid"].attrs["dx"])
            f1 = float(f["params"].attrs["damping_freq_first"])
            f2 = float(f["params"].attrs["damping_freq_second"])
        if dx != 1.0:
            raise ValueError(f"{sid}: dx={dx}, the cache assumes 1 m cells")
        np.savez(
            out / f"{name}__{sid}.npz",
            vs=vs,
            zeta=zeta,
            x_m=lab["x_m"],
            log_tf=np.log(np.maximum(lab["tf"], LOG_FLOOR)).astype(np.float32),
            split=np.array(sp),
            f1_hz=np.float32(f1),
            f2_hz=np.float32(f2),
        )
    print(f"sections {name}: {sids.size} runs, splits {dict(zip(*np.unique(split, return_counts=True)))}", flush=True)
    return freq


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iid-h5-dir", type=Path)
    ap.add_argument("--iid-tf-dir", type=Path)
    ap.add_argument("--iid-limit", type=int, default=None, help="random subset of IID runs (default all)")
    ap.add_argument("--section", action="append", default=[], help="NAME=H5_DIR:LABEL_DIR (repeatable)")
    ap.add_argument("--fixed", action="append", default=[],
                    help="NAME=H5_DIR:TF_DIR:MANIFEST:RECORDER (fixed-window campaign)")
    ap.add_argument("--manifold-h5", type=Path)
    ap.add_argument("--manifold-labels", type=Path)
    ap.add_argument("--manifold-manifest", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    freqs = []
    if args.iid_h5_dir:
        freqs.append(build_iid(args.iid_h5_dir, args.iid_tf_dir, args.out / "iid", args.iid_limit, args.workers))
    for spec in args.fixed:
        name, _, rest = spec.partition("=")
        h5_dir, tf_dir, manifest, recorder = rest.split(":")
        freqs.append(build_fixed_campaign(
            name, Path(h5_dir), Path(tf_dir), Path(manifest), Path(recorder), args.out / name, args.workers,
        ))
    for spec in args.section:
        name, _, rest = spec.partition("=")
        h5_dir, _, label_dir = rest.partition(":")
        freqs.append(build_sections(name, Path(h5_dir), Path(label_dir), args.out / "sections"))
    if args.manifold_h5:
        freqs.append(build_manifold(args.manifold_h5, args.manifold_labels, args.manifold_manifest, args.out / "sections"))
    if not freqs:
        raise SystemExit("nothing to build: pass --iid-h5-dir and/or --section")
    for f in freqs[1:]:
        if not np.allclose(f, freqs[0], rtol=1e-4):
            raise ValueError("IID and section frequency grids differ")
    np.save(args.out / "freq.npy", np.asarray(freqs[0], dtype=np.float32))
    meta = {
        "iid_h5_dir": str(args.iid_h5_dir), "iid_tf_dir": str(args.iid_tf_dir), "sections": args.section,
        "fixed": args.fixed, "manifold_h5": str(args.manifold_h5), "nz_iid": NZ_IID,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
