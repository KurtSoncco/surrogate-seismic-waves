#!/usr/bin/env python3
"""One batch active-learning round over whole sections (plan step 7).

Candidates (AL4PDE pool, ACCEL / frontier-learning edits):
  * single-parameter mutations of the highest-error simulated sections
    (error = ensemble-mean log RMS against the section-window labels in
    0.1-2.5 Hz; sections without labels are ranked by ensemble spread);
  * about 30 % fresh Sobol draws.
Case 7 and its close neighbours are excluded by section_generator.near_case7.

Selection (AL4PDE's two best methods) on the deep-ensemble posterior
features phi(x) = (log f_m(x) - mean_m log f(x)) / sqrt(M - 1), stacked over
members, windows (120 m stride), stations (every 5th) and 24 log-spaced
frequencies in 0.1-2.5 Hz, so k(x, x') = phi . phi' is the ensemble
covariance:
  * ``lcmd``  largest-cluster maximum distance (Holzmueller et al. 2023), with
    the simulated sections as initial centres;
  * ``sbal``  stochastic batch AL: draw without replacement with p ~ var^beta.

    python acquire_sections.py --model-dir CK0 --model-dir CK1 --model-dir CK2 \
        --manifest sections/round0.csv --windows DATA/windows_round0 \
        --batch 12 --round 1 --out sections/round1.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from section_generator import mutate, near_case7, read_rows, sample_rows, section_field, write_rows
from section_windows import STATION_SPACING, WINDOW_NX, window_starts
from score_section import load_models

TRUSTED_HZ = (0.1, 2.5)
FEATURE_WINDOW_STRIDE_M = 120
FEATURE_STATION_STRIDE = 5
N_FEATURE_FREQ = 24
ERROR_WINDOW_STRIDE = 4
RANDOM_SHARE = 0.3


class Ensemble:
    def __init__(self, model_dirs: list[Path], freq_path: Path | None = None):
        self.models, self.device, self.config, self.target = load_models(model_dirs)
        if self.target != "residual":
            raise ValueError("acquisition expects residual-target members (TF / TF_1D)")
        from absolute_inputs import encode_absolute

        self._encode = encode_absolute
        self.rec = np.asarray(self.config.recorder_x_indices())
        freq = np.load(freq_path or self.config.TF_FREQ_PATH)
        band = np.flatnonzero((freq >= TRUSTED_HZ[0]) & (freq <= TRUSTED_HZ[1]))
        self.f_feat = band[np.unique(np.round(np.geomspace(1, band.size, N_FEATURE_FREQ)).astype(int) - 1)]
        self.f_band = band

    def log_residual(self, vs: np.ndarray, zeta: np.ndarray, starts: np.ndarray, batch: int = 8) -> np.ndarray:
        """log(TF / TF_1D) of every member at the 21 stations: (M, n_win, 21, F)."""
        import torch

        c = self.config
        xs = np.stack([
            self._encode(vs[:, s : s + WINDOW_NX], zeta[:, s : s + WINDOW_NX], vs.shape[0], c.NZ_MAX,
                         dz=1.0, dx=1.0, lx=float(c.LX_VARIABILITY), n_prior=int(c.PRIOR_CHANNELS))
            for s in starts
        ])
        out = []
        for m in self.models:
            parts = []
            for i in range(0, len(xs), batch):
                with torch.no_grad():
                    g = m(torch.from_numpy(xs[i : i + batch]).to(self.device)).cpu().numpy()
                parts.append(np.log(np.maximum(g[:, self.rec, :], 1e-6)))
            out.append(np.concatenate(parts))
        return np.stack(out)

    def features(self, vs: np.ndarray, zeta: np.ndarray) -> np.ndarray:
        starts = window_starts(vs.shape[1])[:: FEATURE_WINDOW_STRIDE_M // STATION_SPACING]
        if starts.size == 0:
            starts = np.array([0])
        lr = self.log_residual(vs, zeta, starts)[:, :, ::FEATURE_STATION_STRIDE][..., self.f_feat]
        dev = lr - lr.mean(axis=0, keepdims=True)
        return (dev / math.sqrt(max(len(self.models) - 1, 1))).reshape(-1)


def section_errors(ens: Ensemble, window_dirs: list[Path]) -> dict[str, float]:
    """Ensemble-mean log RMS error per simulated section over its labelled windows."""
    import hdf5plugin  # noqa: F401
    import h5py

    errors: dict[str, list[float]] = defaultdict(list)
    for d in window_dirs:
        rows = list(csv.DictReader(open(d / "manifest.csv")))
        tf = np.load(d / "tf_outcrop_per_sample.npy", mmap_mode="r")
        tf1d = np.load(d / "tf1d_outcrop_per_sample.npy", mmap_mode="r")
        by_run: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(rows):
            by_run[r["h5_path"]].append(i)
        for path, idx in by_run.items():
            idx = idx[::ERROR_WINDOW_STRIDE]
            with h5py.File(path, "r") as f:
                vs = f["Vs_realization_2D"][:].astype(np.float64)
                zeta = f["Damping_zeta"][:].astype(np.float64)
            starts = np.array([int(rows[i]["x_start"]) for i in idx])
            pred = ens.log_residual(vs, zeta, starts).mean(axis=0)
            true = np.log(np.maximum(tf[idx], 1e-6)) - np.log(np.maximum(tf1d[idx], 1e-6))
            err = (pred - true)[..., ens.f_band]
            errors[rows[idx[0]]["group"]].append(float(np.sqrt(np.mean(err**2))))
    return {k: float(np.mean(v)) for k, v in errors.items()}


def lcmd(phi_pool: np.ndarray, phi_train: np.ndarray, k: int) -> list[int]:
    """Largest-cluster maximum distance on the linear kernel of phi.

    Simulated sections are the initial centres; without them the first pick
    is the largest-variance candidate. ``dist < 0`` marks picked candidates.
    """
    sq_pool = np.einsum("ij,ij->i", phi_pool, phi_pool)
    chosen: list[int] = []
    if len(phi_train):
        c = phi_train
        d2 = sq_pool[:, None] + np.einsum("ij,ij->i", c, c)[None, :] - 2 * phi_pool @ c.T
        nearest, dist = d2.argmin(axis=1), np.maximum(d2.min(axis=1), 0.0)
        n_centres = len(c)
    else:
        first = int(np.argmax(sq_pool))
        nearest = np.zeros(len(phi_pool), dtype=int)
        dist = np.maximum(sq_pool + sq_pool[first] - 2 * phi_pool @ phi_pool[first], 0.0)
        dist[first] = -1.0
        chosen.append(first)
        n_centres = 1
    while len(chosen) < k:
        open_mask = dist >= 0
        cluster_size = defaultdict(float)
        for i in np.flatnonzero(open_mask):
            cluster_size[int(nearest[i])] += dist[i]
        best_cluster = max(cluster_size, key=cluster_size.get)
        members = np.flatnonzero(open_mask & (nearest == best_cluster))
        pick = int(members[np.argmax(dist[members])])
        chosen.append(pick)
        new_c = phi_pool[pick]
        d_new = np.maximum(sq_pool + new_c @ new_c - 2 * phi_pool @ new_c, 0.0)
        closer = open_mask & (d_new < dist)
        nearest[closer] = n_centres
        n_centres += 1
        dist[closer] = d_new[closer]
        dist[pick] = -1.0
    return chosen


def sbal(phi_pool: np.ndarray, k: int, rng: np.random.Generator, beta: float = 1.0) -> list[int]:
    var = np.einsum("ij,ij->i", phi_pool, phi_pool) + 1e-12
    p = var**beta / np.sum(var**beta)
    return [int(i) for i in rng.choice(len(var), size=k, replace=False, p=p)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, action="append", required=True)
    ap.add_argument("--manifest", type=Path, action="append", required=True, help="already simulated section rows")
    ap.add_argument("--windows", type=Path, action="append", default=[], help="section_windows.py outputs (labels)")
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--pool-factor", type=int, default=8)
    ap.add_argument("--n-parents", type=int, default=None, help="default: batch // 2")
    ap.add_argument("--method", choices=("lcmd", "sbal"), default="lcmd")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--freq", type=Path, default=None, help="frequency grid (default: the members' GIFNO_TF_DIR/freq.npy)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if len(args.model_dir) < 2:
        raise SystemExit("deep-ensemble acquisition needs at least two members")
    rng = np.random.default_rng([args.seed, args.round])
    ens = Ensemble(args.model_dir, args.freq)

    done = [r for m in args.manifest for r in read_rows(m) if r.get("split", "train") == "train"]
    errors = section_errors(ens, args.windows) if args.windows else {}
    phi_done = {r["sample_id"]: ens.features(*section_field(r)[:2]) for r in done}
    score = {sid: errors.get(sid, float(np.linalg.norm(phi))) for sid, phi in phi_done.items()}
    n_parents = args.n_parents or max(1, args.batch // 2)
    parents = sorted(done, key=lambda r: -score[r["sample_id"]])[:n_parents]
    print("parents:", [(p["sample_id"], round(score[p["sample_id"]], 4)) for p in parents])

    pool_n = args.pool_factor * args.batch
    n_rand = int(round(RANDOM_SHARE * pool_n))
    tag = f"r{args.round:02d}"
    pool = [mutate(parents[i % len(parents)], rng, f"{tag}m_{i:04d}") for i in range(pool_n - n_rand)]
    pool += sample_rows(n_rand, seed=int(rng.integers(1 << 31)), prefix=f"{tag}u")
    pool = [r for r in pool if not near_case7(r)]
    phi_pool = np.stack([ens.features(*section_field(r)[:2]) for r in pool])
    phi_train = np.stack(list(phi_done.values())) if phi_done else np.empty((0, phi_pool.shape[1]))
    k = min(args.batch, len(pool))
    picks = lcmd(phi_pool, phi_train, k) if args.method == "lcmd" else sbal(phi_pool, k, rng)

    out = []
    for j, i in enumerate(picks):
        r = dict(pool[i])
        r["acq_round"] = args.round
        r["acq_method"] = args.method
        r["acq_order"] = j
        r["ensemble_var"] = float(phi_pool[i] @ phi_pool[i])
        r["origin"] = "mutation" if "parent_id" in r else "random"
        out.append(r)
    write_rows(out, args.out)
    n_mut = sum(r["origin"] == "mutation" for r in out)
    print(f"round {args.round}: {len(out)} sections ({n_mut} mutations, {len(out) - n_mut} random) -> {args.out}")


if __name__ == "__main__":
    main()
