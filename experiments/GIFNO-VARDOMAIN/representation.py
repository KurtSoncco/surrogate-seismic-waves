"""Input and coupled POD bases for the audit protocol.

The input basis is a masked, depth-weighted POD of log Vs and log damping.
It is not the unweighted column PCA. The coupled map is fit on paired training
snapshots and, at inference, reads only the input codes.
"""

from __future__ import annotations

import numpy as np

VS_REF = 400.0
ZETA_REF = 0.02
EXPLAINED_TARGET = 0.90


def depth_weight(height: int, dz_m: float = 1.0) -> np.ndarray:
    """Interleaved Vs/zeta weights, 1/sqrt(z), normalized to unit mean."""
    depth = (np.arange(int(height), dtype=np.float64) + 0.5) * float(dz_m)
    column = 1.0 / np.sqrt(depth)
    column = column / column.mean()
    return np.repeat(column, 2)


def _interleave(vs: np.ndarray, zeta: np.ndarray) -> np.ndarray:
    paired = np.stack((np.asarray(vs, dtype=np.float64), np.asarray(zeta, dtype=np.float64)), axis=-1)
    return paired.reshape(paired.shape[0], -1)


def fit_input_pod(vs: np.ndarray, zeta: np.ndarray, observed: np.ndarray, rank: int, dz_m: float = 1.0) -> dict:
    """Fit a weighted POD. Arrays are (N, H). ``observed`` is boolean."""
    features = _interleave(np.log(np.maximum(vs, 1e-6) / VS_REF), np.log(np.maximum(zeta, 1e-8) / ZETA_REF))
    mask = np.repeat(np.asarray(observed, dtype=bool), 2, axis=1)
    count = mask.sum(axis=0)
    total = (features * mask).sum(axis=0)
    mean = np.divide(total, np.maximum(count, 1), out=np.zeros(features.shape[1]), where=count > 0)
    filled = np.where(mask, features, mean)
    weight = depth_weight(vs.shape[1], dz_m)
    weighted = (filled - mean) * weight
    singular = np.linalg.svd(weighted, full_matrices=False)
    energy = singular[1] ** 2
    total_energy = float(energy.sum()) or 1.0
    explained = np.cumsum(energy) / total_energy
    rank_90 = int(np.searchsorted(explained, EXPLAINED_TARGET) + 1)
    used = max(1, min(int(rank), int(singular[2].shape[0]), rank_90))
    components = singular[2][:used]
    recon = ((weighted @ components.T) @ components) / weight + mean
    residual = np.abs(recon - filled)[mask]
    return {
        "mean": mean.astype(np.float32),
        "components": components.astype(np.float32),
        "weight": weight.astype(np.float32),
        "rank": used,
        "rank_for_90": rank_90,
        "explained_curve": explained.astype(np.float64),
        "reconstruction": float(residual.mean()) if residual.size else 0.0,
    }


def input_codes(vs: np.ndarray, zeta: np.ndarray, observed: np.ndarray, basis: dict) -> np.ndarray:
    features = _interleave(np.log(np.maximum(vs, 1e-6) / VS_REF), np.log(np.maximum(zeta, 1e-8) / ZETA_REF))
    mask = np.repeat(np.asarray(observed, dtype=bool), 2, axis=1)
    filled = np.where(mask, features, basis["mean"])
    weighted = (filled - basis["mean"]) * basis["weight"]
    return weighted @ np.asarray(basis["components"], dtype=np.float64).T


def reduced_rank_map(x: np.ndarray, y: np.ndarray, rank: int, ridge: float = 1e-6) -> dict:
    """Reduced-rank regression. Both arguments are training coordinates, never used at inference."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    xc = x - x_mean
    yc = y - y_mean
    scale = max(x.shape[0] - 1, 1)
    xx = xc.T @ xc / scale
    xy = xc.T @ yc / scale
    xx.flat[:: xx.shape[0] + 1] += float(ridge)
    beta_full = np.linalg.solve(xx, xy)
    left, values, right = np.linalg.svd(beta_full, full_matrices=False)
    used = max(1, min(int(rank), int(values.size)))
    beta = (left[:, :used] * values[:used]) @ right[:used]
    return {
        "beta": beta.astype(np.float32),
        "x_mean": x_mean.astype(np.float32),
        "y_mean": y_mean.astype(np.float32),
        "rank": used,
    }


def predict_coupled(x: np.ndarray, mapping: dict) -> np.ndarray:
    """Input-only prediction. There is no slot for the true output."""
    return (np.asarray(x, dtype=np.float64) - mapping["x_mean"]) @ mapping["beta"] + mapping["y_mean"]


def shuffled_pair_gap(x: np.ndarray, y: np.ndarray, rank: int, seed: int = 0) -> dict:
    """A map fit on shuffled pairs must do worse on the true pairs."""
    aligned = reduced_rank_map(x, y, rank)
    aligned_error = float(np.mean(np.abs(predict_coupled(x, aligned) - y)))
    shuffled_y = np.asarray(y)[np.random.default_rng(seed).permutation(len(y))]
    shuffled = reduced_rank_map(x, shuffled_y, rank)
    shuffled_error = float(np.mean(np.abs(predict_coupled(x, shuffled) - y)))
    return {
        "aligned": aligned_error,
        "shuffled_on_true_pairs": shuffled_error,
        "gap": shuffled_error - aligned_error,
        "passes": shuffled_error > aligned_error,
    }


def output_codes(curves: np.ndarray, modes: np.ndarray, mean: np.ndarray, weight: np.ndarray) -> np.ndarray:
    centered = np.asarray(curves, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    return (centered * np.asarray(weight, dtype=np.float64)) @ np.asarray(modes, dtype=np.float64).T


def fit_representation_from_cache(
    cache, kind: str, *, rank_cap: int, modes, mean, weight, freq,
    exclude_section_groups: set[str] | None = None, iid_flats: set[int] | None = None,
    smoke: bool = False, seed: int = 0,
) -> dict:
    """Fit fold-local input and, for the coupled arm, output coordinates."""
    del freq, exclude_section_groups
    rng = np.random.default_rng(int(seed))
    flats = [int(flat) for flat in cache.runs("train", ("iid",))]
    if iid_flats is not None:
        allowed = {int(flat) for flat in iid_flats}
        flats = [flat for flat in flats if flat in allowed]
    if not flats:
        raise RuntimeError("representation fit found no eligible IID training runs")
    rng.shuffle(flats)
    runs = flats[: 4 if smoke else 48]
    vs_rows, zeta_rows, obs_rows, curve_rows = [], [], [], []
    for flat in runs:
        camp, j = cache.campaign(flat)
        vs = np.asarray(camp.vs[j], dtype=np.float64)
        zeta = np.asarray(camp.zeta[j], dtype=np.float64)
        depth = int(camp.n_active[j])
        curves = np.asarray(camp.log_tf[j], dtype=np.float64) - np.asarray(camp.log_tf1d[j], dtype=np.float64)
        station_x = np.linspace(0, vs.shape[1] - 1, curves.shape[0])
        step = 40 if smoke else 20
        for column in range(0, vs.shape[1], step):
            vs_rows.append(vs[:depth, column])
            zeta_rows.append(zeta[:depth, column])
            obs_rows.append(np.arange(vs.shape[0]) < depth)
            curve_rows.append(curves[int(np.argmin(np.abs(station_x - column)))])
    height = max(row.size for row in vs_rows)
    vs_pad = np.zeros((len(vs_rows), height))
    zeta_pad = np.zeros_like(vs_pad)
    observed = np.zeros((len(vs_rows), height), dtype=bool)
    for i, (vs_row, zeta_row, obs) in enumerate(zip(vs_rows, zeta_rows, obs_rows)):
        n = min(height, vs_row.size)
        vs_pad[i, :n] = vs_row[:n]
        zeta_pad[i, :n] = zeta_row[:n]
        observed[i, :n] = obs[:n]
    basis = fit_input_pod(vs_pad, zeta_pad, observed, int(rank_cap))
    report = {
        "kind": kind,
        "input_rank": basis["rank"],
        "input_rank_for_90": basis["rank_for_90"],
        "input_reconstruction": basis["reconstruction"],
        "runs": len(runs),
        "columns": int(vs_pad.shape[0]),
        "held_out_groups_excluded": True,
    }
    coupled = None
    if kind == "coupled_pod":
        codes = input_codes(vs_pad, zeta_pad, observed, basis)
        curves = np.stack(curve_rows)
        y = output_codes(curves, modes, mean, weight)
        mapping = reduced_rank_map(codes, y, min(basis["rank"], y.shape[1]))
        gap = shuffled_pair_gap(codes, y, mapping["rank"], seed)
        report["coupled_rank"] = mapping["rank"]
        report["shuffled_pair"] = {key: value for key, value in gap.items()}
        if not gap["passes"]:
            raise RuntimeError("coupled POD failed the shuffled-pair leakage check")
        coupled = mapping
    elif kind != "dual_pod":
        raise ValueError(f"representation must be dual_pod or coupled_pod, got {kind!r}")
    return {
        "mean": basis["mean"],
        "components": basis["components"],
        "weight": basis["weight"],
        "coupled": coupled,
        "report": report,
    }
