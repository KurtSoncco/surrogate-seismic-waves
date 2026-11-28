#!/usr/bin/env python3
"""Does a depth code predict the 1D transfer function better than a linear basis?

The frequency autoencoder lost to POD, so this probe does not touch it. The
target is log|TF_1D| at each receiver: that curve is the response of the
receiver's own column run as laterally uniform. Three representations of that
column are compared with one linear readout:

- the frozen 32-wide column autoencoder
- a 16-component masked PCA of log Vs and log zeta
- four physical features: quarter-wavelength f0, thickness, top-10 m Vs, mean zeta

PCA and POD are fit on training runs only. The readout is fit on the training
receivers and scored on the validation runs. Rows past n_active are missing.
The pairwise covariance uses that suffix pattern, and a short column is scored
by filling the missing rows with the training mean.

    python depth_probe.py --cache ... --ae .../depth_ae.pt --out .../depth_probe
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import Cache
from model import ColumnDepthAutoencoder
from pretrain_ae import DEPTH_REF_M, VS_REF, ZETA_REF, _refuse_shipped, _strip_batch
from screen_hifreq import BANDS, band_means

PCA_RANK = 16
POD_RANK = 64
RIDGE = 1.0
NEAR_SURFACE_M = 10
DZ_M = 1.0
CLEAR_MARGIN = 0.10
TIE_MARGIN = 0.05


def quarter_wavelength_hz(vs: np.ndarray) -> np.ndarray:
    """f0 = 1 / (4 * travel time) for a soil column over a much stiffer base."""
    travel = np.sum(DZ_M / np.maximum(vs, 1e-6), axis=0)
    return 1.0 / (4.0 * travel)


def physical_features(vs: np.ndarray, zeta: np.ndarray) -> np.ndarray:
    """(W, 4) logs of f0, thickness, near-surface Vs, and mean damping. No curve is used."""
    height = int(vs.shape[0])
    top = min(NEAR_SURFACE_M, height)
    raw = np.stack(
        [
            quarter_wavelength_hz(vs),
            np.full(vs.shape[1], height * DZ_M, dtype=np.float64),
            vs[:top].mean(axis=0),
            zeta.mean(axis=0),
        ],
        axis=1,
    )
    return np.log(np.maximum(raw, 1e-8))


def receiver_curves(camp, j: int, n_freq: int) -> tuple[np.ndarray, np.ndarray]:
    """Columns under the receivers, and their log|TF_1D| curves shaped (stations, freq)."""
    cols = np.asarray(camp.rec_x, dtype=np.int64)
    raw = np.asarray(camp.log_tf1d[j], dtype=np.float64)
    if raw.shape == (cols.size, n_freq):
        return cols, raw
    if raw.shape == (n_freq, cols.size):
        return cols, raw.T
    raise RuntimeError(f"log_tf1d[{j}] has shape {raw.shape}, expected {(cols.size, n_freq)}")


def _pair_features(values: torch.Tensor) -> np.ndarray:
    """(2, H, W) log channels to (W, 2H) with feature 2*row+channel."""
    vs = values[0].detach().cpu().numpy().astype(np.float64)
    zeta = values[1].detach().cpu().numpy().astype(np.float64)
    width = vs.shape[1]
    paired = np.empty((width, vs.shape[0] * 2), dtype=np.float64)
    paired[:, 0::2] = vs.T
    paired[:, 1::2] = zeta.T
    return paired


def accumulate_gram(
    cross: np.ndarray, cross_n: np.ndarray, total: np.ndarray, count: np.ndarray, values: torch.Tensor,
) -> None:
    paired = _pair_features(values[0] if values.ndim == 4 else values)
    width = paired.shape[1]
    cross[:width, :width] += paired.T @ paired
    cross_n[:width, :width] += paired.shape[0]
    total[:width] += paired.sum(axis=0)
    count[:width] += paired.shape[0]


def masked_pca(cross: np.ndarray, cross_n: np.ndarray, total: np.ndarray, count: np.ndarray, rank: int) -> dict:
    """Pairwise-complete covariance. Missing rows never enter a pair they do not both have."""
    observed = count > 0
    mean = np.zeros_like(total)
    mean[observed] = total[observed] / count[observed]
    both = cross_n > 1
    centered = cross - (mean[:, None] * mean[None, :]) * cross_n
    cov = np.zeros_like(cross)
    cov[both] = centered[both] / (cross_n[both] - 1.0)
    cov = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    positive = vals > 0
    taken = min(rank, int(positive.sum()))
    components = vecs[:, :taken].T
    explained = float(vals[:taken].clip(min=0).sum() / max(vals.clip(min=0).sum(), 1e-12))
    return {
        "mean": mean,
        "components": components,
        "explained": explained,
        "negative_eigenvalues": int((vals < -1e-8).sum()),
        "rank": int(taken),
    }


def project_columns(values: torch.Tensor, pca: dict) -> np.ndarray:
    paired = _pair_features(values[0] if values.ndim == 4 else values)
    full = np.repeat(pca["mean"][None, :], paired.shape[0], axis=0)
    full[:, : paired.shape[1]] = paired
    centered = full - pca["mean"]
    return centered @ pca["components"].T


def profile_l1(values: torch.Tensor, pca: dict, scores: np.ndarray) -> float:
    paired = _pair_features(values[0] if values.ndim == 4 else values)
    recon = scores @ pca["components"] + pca["mean"]
    err = np.abs(recon[:, : paired.shape[1]] - paired)
    return float(err.mean()) if err.size else 0.0


def fit_pod(cross: np.ndarray, total: np.ndarray, count: int, rank: int) -> dict:
    mean = total / max(count, 1)
    cov = (cross - count * np.outer(mean, mean)) / max(count - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    taken = min(rank, int((vals > 0).sum()))
    modes = vecs[:, order][:, :taken].T
    explained = float(vals[:taken].clip(min=0).sum() / max(vals.clip(min=0).sum(), 1e-12))
    return {"mean": mean, "modes": modes, "explained": explained, "rank": int(taken)}


def pod_coefficients(curves: np.ndarray, pod: dict) -> np.ndarray:
    return (curves - pod["mean"]) @ pod["modes"].T


def reconstruct(coeff: np.ndarray, pod: dict) -> np.ndarray:
    return pod["mean"] + coeff @ pod["modes"]


class NormalEquations:
    def __init__(self, width: int, targets: int):
        self.sum_x = np.zeros(width, dtype=np.float64)
        self.sum_xx = np.zeros((width, width), dtype=np.float64)
        self.sum_xy = np.zeros((width, targets), dtype=np.float64)
        self.sum_y = np.zeros(targets, dtype=np.float64)
        self.n = 0

    def add(self, x: np.ndarray, y: np.ndarray) -> None:
        self.sum_x += x.sum(axis=0)
        self.sum_xx += x.T @ x
        self.sum_xy += x.T @ y
        self.sum_y += y.sum(axis=0)
        self.n += int(x.shape[0])

    def solve(self, alpha: float) -> dict:
        mean = self.sum_x / self.n
        centered = self.sum_xx - self.n * np.outer(mean, mean)
        scale = np.sqrt(np.maximum(np.diag(centered) / max(self.n - 1, 1), 1e-12))
        inv = 1.0 / scale
        ztz = centered * inv[:, None] * inv[None, :]
        zty = (self.sum_xy - mean[:, None] * self.sum_y[None, :]) * inv[:, None]
        beta = np.linalg.solve(ztz + alpha * np.eye(self.sum_x.size), zty)
        return {"mean": mean, "scale": scale, "beta": beta, "y_mean": self.sum_y / self.n, "n": self.n}

    def predict(self, x: np.ndarray, solution: dict) -> np.ndarray:
        z = (x - solution["mean"]) / solution["scale"]
        return solution["y_mean"] + z @ solution["beta"]


def curve_scores(pred: np.ndarray, target: np.ndarray, freq: np.ndarray) -> dict:
    per_bin = np.abs(pred - target).mean(axis=0)
    bands = band_means(freq, {"error": per_bin})
    scale = float(np.linalg.norm(target))
    return {
        "mae": float(per_bin.mean()),
        "frobenius": float(np.linalg.norm(pred - target) / scale) if scale else 0.0,
        "bands": {key: float(row["error"]) for key, row in bands.items()},
    }


def decide(scores: dict[str, dict]) -> str:
    """PCA wins ties. A learned code is kept only when it is clearly ahead. Physical features can retire both."""
    ae = scores["autoencoder"]["mae"]
    pca = scores["pca"]["mae"]
    physical = scores["physical"]["mae"]
    learned = min(ae, pca)

    def close(left: float, right: float) -> bool:
        return abs(left - right) <= TIE_MARGIN * max(left, right, 1e-8)

    if close(physical, learned):
        return "physical"
    if ae < pca * (1.0 - CLEAR_MARGIN):
        return "autoencoder"
    return "pca"


DECISION_TEXT = {
    "physical": "Physical features match the learned codes. The 1D response does not need a depth autoencoder or a PCA code.",
    "autoencoder": "The column autoencoder is clearly ahead of masked PCA on the reconstructed 1D spectrum.",
    "pca": "Masked PCA ties or beats the column autoencoder. Use the PCA code.",
}


def _ae_codes(model: ColumnDepthAutoencoder, values: torch.Tensor, mask: torch.Tensor, depth: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        surface, _recon = model(values, mask, depth)
    return surface[0, :, 0, :].detach().cpu().numpy().T.astype(np.float64)


def _iter_runs(cache: Cache, split: str):
    for flat in cache.runs(split, ("iid",)):
        camp, j = cache.campaign(int(flat))
        yield int(flat), camp, j


def _section_columns(sec, height: int, stride: int) -> torch.Tensor | None:
    """(1, 2, rows, columns) log features of one manifold section, rows capped at ``height``."""
    from pretrain_ae import VS_REF, ZETA_REF

    rows = min(int(sec.vs.shape[0]), int(height))
    cols = np.arange(0, int(sec.vs.shape[1]), max(int(stride), 1))
    if rows < 1 or cols.size < 1:
        return None
    vs = np.asarray(sec.vs[:rows][:, cols], dtype=np.float64)
    zeta = np.asarray(sec.zeta[:rows][:, cols], dtype=np.float64)
    values = np.empty((1, 2, rows, cols.size), dtype=np.float32)
    values[0, 0] = np.log(np.maximum(vs, 1e-6) / VS_REF)
    values[0, 1] = np.log(np.maximum(zeta, 1e-8) / ZETA_REF)
    return torch.from_numpy(values)


def fit_column_pca(
    cache: Cache, rank: int = PCA_RANK, *, joint: bool = False, column_stride: int = 10,
    exclude_section_groups: set[str] | None = None, iid_groups: set[int] | None = None,
) -> dict:
    """Masked PCA of training columns. The basis is fit once and then frozen.

    ``joint`` fits one basis on IID and manifold columns, padded to the metric
    canvas depth. Rows a column does not have stay missing and are zeroed in
    the components, so a 150 m or 250 m profile never addresses an untrained row
    by index: the missing suffix is the training mean.
    """
    from encoding import NZ_METRIC

    if joint:
        height = NZ_METRIC
    else:
        height = 0
        for _flat, camp, j in _iter_runs(cache, "train"):
            height = max(height, int(camp.n_active[j]))
    if height < 1:
        raise RuntimeError("no training IID columns for the depth PCA")
    features = 2 * height
    cross = np.zeros((features, features), dtype=np.float64)
    cross_n = np.zeros((features, features), dtype=np.float64)
    total = np.zeros(features, dtype=np.float64)
    count = np.zeros(features, dtype=np.float64)
    runs = 0
    for flat, camp, j in _iter_runs(cache, "train"):
        if iid_groups is not None and int(camp.group[j]) not in iid_groups:
            continue
        width = int(cache.campaign(flat)[0].vs.shape[-1])
        step = max(int(column_stride), 1) if joint else 1
        packed = _strip_batch(cache, flat, list(range(0, width, step)))
        if packed is None:
            continue
        accumulate_gram(cross, cross_n, total, count, packed[0])
        runs += 1
    sections = 0
    if joint:
        for sec in cache.sections:
            if sec.split != "train":
                continue
            if exclude_section_groups and sec.group_id in exclude_section_groups:
                continue
            packed = _section_columns(sec, height, column_stride)
            if packed is None:
                continue
            accumulate_gram(cross, cross_n, total, count, packed)
            sections += 1
    pca = masked_pca(cross, cross_n, total, count, rank)
    unseen = count <= 0
    if np.any(unseen):
        pca["components"] = np.array(pca["components"], copy=True)
        pca["components"][:, unseen] = 0.0
    pca["runs"] = runs
    pca["sections"] = sections
    pca["height"] = height
    pca["joint"] = bool(joint)
    return pca


def fit_bases(cache: Cache) -> tuple[dict, dict, int]:
    height = 0
    for flat, camp, j in _iter_runs(cache, "train"):
        height = max(height, int(camp.n_active[j]))
    features = 2 * height
    cross = np.zeros((features, features), dtype=np.float64)
    cross_n = np.zeros((features, features), dtype=np.float64)
    total = np.zeros(features, dtype=np.float64)
    count = np.zeros(features, dtype=np.float64)
    freq = int(np.asarray(cache.freq).size)
    pod_cross = np.zeros((freq, freq), dtype=np.float64)
    pod_sum = np.zeros(freq, dtype=np.float64)
    pod_n = 0
    runs = 0
    for flat, camp, j in _iter_runs(cache, "train"):
        width = int(camp.vs.shape[-1])
        packed = _strip_batch(cache, flat, list(range(width)))
        if packed is None:
            continue
        values, _mask, _depth = packed
        accumulate_gram(cross, cross_n, total, count, values)
        _cols, curves = receiver_curves(camp, j, freq)
        pod_cross += curves.T @ curves
        pod_sum += curves.sum(axis=0)
        pod_n += int(curves.shape[0])
        runs += 1
        if runs % 1000 == 0:
            print(f"basis runs {runs}", flush=True)
    pca = masked_pca(cross, cross_n, total, count, PCA_RANK)
    pod = fit_pod(pod_cross, pod_sum, pod_n, POD_RANK)
    print(
        f"pca rank {pca['rank']} explained {pca['explained']:.3f} "
        f"negative eigenvalues {pca['negative_eigenvalues']}",
        flush=True,
    )
    print(f"pod rank {pod['rank']} explained {pod['explained']:.3f} curves {pod_n}", flush=True)
    return pca, pod, pod_n


def f0_peak_correlation(cache: Cache, limit: int = 200) -> float:
    """Correlation of the travel-time f0 with the peak of the stored 1D curve. A low value means the columns are misaligned."""
    freq = np.asarray(cache.freq, dtype=np.float64)
    logs_f0: list[np.ndarray] = []
    logs_peak: list[np.ndarray] = []
    for _seen, (_flat, camp, j) in enumerate(_iter_runs(cache, "train")):
        if _seen >= limit:
            break
        cols, curves = receiver_curves(camp, j, int(freq.size))
        vs = np.asarray(camp.vs[j, : int(camp.n_active[j])][:, cols], dtype=np.float64)
        logs_f0.append(np.log(quarter_wavelength_hz(vs)))
        logs_peak.append(np.log(freq[np.argmax(curves, axis=1)]))
    left = np.concatenate(logs_f0)
    right = np.concatenate(logs_peak)
    return float(np.corrcoef(left, right)[0, 1])


def fit_readouts(cache: Cache, model: ColumnDepthAutoencoder, pca: dict, pod: dict) -> dict[str, NormalEquations]:
    targets = int(pod["modes"].shape[0])
    eqs = {
        "autoencoder": NormalEquations(int(model.code_dim), targets),
        "pca": NormalEquations(int(pca["components"].shape[0]), targets),
        "physical": NormalEquations(4, targets),
    }
    freq = int(np.asarray(cache.freq).size)
    runs = 0
    for flat, camp, j in _iter_runs(cache, "train"):
        cols, curves = receiver_curves(camp, j, freq)
        packed = _strip_batch(cache, flat, cols.tolist())
        if packed is None:
            continue
        values, mask, depth = packed
        coeff = pod_coefficients(curves, pod)
        eqs["autoencoder"].add(_ae_codes(model, values, mask, depth), coeff)
        eqs["pca"].add(project_columns(values, pca), coeff)
        vs = np.asarray(camp.vs[j, : values.shape[2]][:, cols], dtype=np.float64)
        zeta = np.asarray(camp.zeta[j, : values.shape[2]][:, cols], dtype=np.float64)
        eqs["physical"].add(physical_features(vs, zeta), coeff)
        runs += 1
        if runs % 1000 == 0:
            print(f"readout runs {runs}", flush=True)
    return eqs


def score_split(
    cache: Cache,
    split: str,
    model: ColumnDepthAutoencoder,
    pca: dict,
    pod: dict,
    solutions: dict[str, dict],
    equations: dict[str, NormalEquations],
) -> dict:
    freq = np.asarray(cache.freq, dtype=np.float64)
    buckets: dict[str, list[np.ndarray]] = {name: [] for name in ("mean", *solutions)}
    profile = {"autoencoder": [0.0, 0.0], "pca": [0.0, 0.0]}
    runs = 0
    for flat, camp, j in _iter_runs(cache, split):
        cols, curves = receiver_curves(camp, j, int(freq.size))
        packed = _strip_batch(cache, flat, cols.tolist())
        if packed is None:
            continue
        values, mask, depth = packed
        codes = {
            "autoencoder": _ae_codes(model, values, mask, depth),
            "pca": project_columns(values, pca),
        }
        vs = np.asarray(camp.vs[j, : values.shape[2]][:, cols], dtype=np.float64)
        zeta = np.asarray(camp.zeta[j, : values.shape[2]][:, cols], dtype=np.float64)
        codes["physical"] = physical_features(vs, zeta)
        buckets["mean"].append(np.repeat(pod["mean"][None, :], curves.shape[0], axis=0))
        for name, solution in solutions.items():
            pred = reconstruct(equations[name].predict(codes[name], solution), pod)
            buckets[name].append(pred)
        with torch.no_grad():
            _surface, recon = model(values, mask, depth)
        ae_err = (recon - values).abs()
        profile["autoencoder"][0] += float(ae_err.sum())
        profile["autoencoder"][1] += float(mask.sum() * recon.shape[1])
        profile["pca"][0] += profile_l1(values, pca, codes["pca"]) * float(mask.sum() * 2)
        profile["pca"][1] += float(mask.sum() * 2)
        target_bucket = buckets.setdefault("_target", [])
        target_bucket.append(curves)
        runs += 1
    target = np.concatenate(buckets.pop("_target"), axis=0)
    scored = {name: curve_scores(np.concatenate(rows, axis=0), target, freq) for name, rows in buckets.items()}
    for name, (numer, denom) in profile.items():
        scored[name]["profile_masked_l1"] = numer / max(denom, 1.0)
    scored["n_curves"] = int(target.shape[0])
    scored["n_runs"] = runs
    return scored


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--ae", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--code-dim", type=int, default=32)
    args = ap.parse_args()
    _refuse_shipped(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    cache = Cache(args.cache)
    model = ColumnDepthAutoencoder(2, code_dim=args.code_dim).to(device)
    model.load_state_dict(torch.load(args.ae, map_location=device, weights_only=True))
    model.eval()
    print(f"ae code {model.code_dim} device {device}", flush=True)
    aligned = f0_peak_correlation(cache)
    print(f"f0 vs 1D-peak correlation {aligned:.3f}", flush=True)
    if not np.isfinite(aligned) or aligned < 0.3:
        raise SystemExit(f"receiver columns do not line up with log|TF_1D| (correlation {aligned:.3f})")
    pca, pod, n_train_curves = fit_bases(cache)
    equations = fit_readouts(cache, model, pca, pod)
    solutions = {name: eq.solve(RIDGE) for name, eq in equations.items()}
    for name, eq in equations.items():
        print(f"readout {name} columns {eq.n}", flush=True)
    val = score_split(cache, "val", model, pca, pod, solutions, equations)
    choice = decide({name: val[name] for name in ("autoencoder", "pca", "physical")})
    report = {
        "target": "log|TF_1D| at each receiver column, the laterally uniform response of that column",
        "holdout": "validation IID runs",
        "pca_rank": pca["rank"],
        "pca_variance_explained": pca["explained"],
        "pca_negative_eigenvalues": pca["negative_eigenvalues"],
        "pod_rank": pod["rank"],
        "pod_variance_explained": pod["explained"],
        "n_train_curves": n_train_curves,
        "ridge_alpha": RIDGE,
        "features": {
            "f0_hz": "1/(4*sum(dz/Vs)) over the soil",
            "H_m": "n_active * 1 m",
            "vs_near_m_s": f"mean Vs of the top {NEAR_SURFACE_M} m",
            "zeta_mean": "mean zeta over the soil",
            "transform": "natural log of each feature",
        },
        "depth_reference_m": DEPTH_REF_M,
        "vs_ref": VS_REF,
        "zeta_ref": ZETA_REF,
        "clear_margin": CLEAR_MARGIN,
        "tie_margin": TIE_MARGIN,
        "scores": val,
        "decision": choice,
        "decision_text": DECISION_TEXT[choice],
    }
    (args.out / "report.json").write_text(json.dumps(report))
    print(report["decision_text"], flush=True)
    for name in ("mean", "physical", "pca", "autoencoder"):
        row = val[name]
        bands = " ".join(f"{key} {value:.4f}" for key, value in row["bands"].items())
        print(f"{name} mae {row['mae']:.4f} L2 {row['frobenius']:.4f} {bands}", flush=True)


if __name__ == "__main__":
    main()
