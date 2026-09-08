"""Wave 1 inductive-bias instruments (ckpt + fields; unit-testable without).

Prior-collapse, two-group invariance orbits, represent-vs-use probes, and
DCT transfer ratios. Pack leftover DCT also lives in ``gino_bias`` (Wave 0).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

from response_variability.gino_bias import FNO_FREQ_MODES, _EPS, dct_transfer_ratio
from response_variability.metrics import theoretical_f0

PredictR = Callable[[dict[str, np.ndarray]], np.ndarray]


def scale_all_velocities(
    vs_field: np.ndarray,
    vs1: np.ndarray,
    vs2: np.ndarray,
    c_v: float,
) -> dict[str, np.ndarray]:
    """Group 1: multiply every velocity. Impedance ratios unchanged; f0 → c_v f0."""
    return {
        "vs_field": np.asarray(vs_field, dtype=np.float64) * float(c_v),
        "vs1": np.asarray(vs1, dtype=np.float64) * float(c_v),
        "vs2": np.asarray(vs2, dtype=np.float64) * float(c_v),
        "c_v": np.asarray(c_v, dtype=float),
    }


def scale_all_lengths(
    *,
    H: np.ndarray,
    rH: np.ndarray,
    recorder_x: np.ndarray,
    bedrock_thickness: float | np.ndarray,
    domain_length: float | np.ndarray,
    vs2_layer: np.ndarray | None = None,
    c_l: float,
) -> dict[str, np.ndarray]:
    """Group 2: every length. Similarity only if rH, recorders, PML/domain scale too."""
    c = float(c_l)
    out = {
        "H": np.asarray(H, dtype=np.float64) * c,
        "rH": np.asarray(rH, dtype=np.float64) * c,
        "recorder_x": np.asarray(recorder_x, dtype=np.float64) * c,
        "bedrock_thickness": np.asarray(bedrock_thickness, dtype=np.float64) * c,
        "domain_length": np.asarray(domain_length, dtype=np.float64) * c,
        "c_l": np.asarray(c, dtype=float),
    }
    if vs2_layer is not None:
        out["vs2_layer"] = np.asarray(vs2_layer, dtype=np.float64) * c
    return out


def dimensionless_freq(freq: np.ndarray, vs1: float, H: float) -> np.ndarray:
    f0 = theoretical_f0(vs1, H)
    return np.asarray(freq, dtype=float) / max(f0, _EPS)


def impedance_ratio(vs1: np.ndarray, vs2: np.ndarray) -> np.ndarray:
    return np.asarray(vs2, dtype=float) / np.clip(np.asarray(vs1, dtype=float), _EPS, None)


def invariance_orbit_check(
    vs1: np.ndarray,
    vs2: np.ndarray,
    H: np.ndarray,
    *,
    c_v: float = 1.2,
    c_l: float = 1.2,
) -> dict[str, Any]:
    """Sanity: c_v preserves impedance; (c_v, c_l) maps f0 by c_v/c_l.

    The old α:(Vs1,H)→α(Vs1,H) at fixed Vs2 is *not* an invariant (impedance moves).
    """
    z0 = impedance_ratio(vs1, vs2)
    vel = scale_all_velocities(vs1, vs1, vs2, c_v)  # field unused; vs1 placeholder
    z_v = impedance_ratio(vel["vs1"], vel["vs2"])
    lengths = scale_all_lengths(
        H=H,
        rH=np.full_like(H, 40.0, dtype=float),
        recorder_x=np.linspace(0.0, 500.0, 21),
        bedrock_thickness=10.0,
        domain_length=500.0,
        c_l=c_l,
    )
    f0_0 = np.array([theoretical_f0(v, h) for v, h in zip(vs1, H)])
    f0_v = np.array([theoretical_f0(v, h) for v, h in zip(vel["vs1"], H)])
    f0_l = np.array([theoretical_f0(v, h) for v, h in zip(vs1, lengths["H"])])
    # Broken α map
    alpha = c_v
    z_alpha = impedance_ratio(vs1 * alpha, vs2)
    return {
        "impedance_preserved_cv": bool(np.allclose(z0, z_v, rtol=1e-12, atol=1e-12)),
        "f0_scales_cv": bool(np.allclose(f0_v, f0_0 * c_v, rtol=1e-12, atol=1e-10)),
        "f0_scales_cl": bool(np.allclose(f0_l, f0_0 / c_l, rtol=1e-12, atol=1e-10)),
        "alpha_vs1_H_breaks_impedance": bool(
            np.max(np.abs(z_alpha - z0) / np.clip(z0, _EPS, None)) > 0.05
        ),
        "c_v": float(c_v),
        "c_l": float(c_l),
    }


def mean_r_on_train(r: np.ndarray) -> dict[str, float]:
    x = np.asarray(r, dtype=np.float64)
    return {
        "mean_R": float(np.nanmean(x)),
        "mean_abs_R": float(np.nanmean(np.abs(x))),
        "near_zero": bool(abs(float(np.nanmean(x))) < 0.05 * max(float(np.nanmean(np.abs(x))), _EPS)),
    }


def prior_collapse_curve(
    distances: np.ndarray,
    r_norm: np.ndarray,
    *,
    hull: float = 0.0,
) -> dict[str, Any]:
    """Describe R̂ magnitude vs distance past the train hull (no labels).

    Cliff = drop concentrated at the boundary; smooth = gradual decay.
    """
    d = np.asarray(distances, dtype=float)
    y = np.asarray(r_norm, dtype=float)
    order = np.argsort(d)
    d, y = d[order], y[order]
    inside = y[d <= hull + 1e-9] if np.any(d <= hull + 1e-9) else y[: max(len(y) // 5, 1)]
    far = y[d >= np.quantile(d, 0.8)] if d.size >= 5 else y[-max(len(y) // 5, 1) :]
    y_in = float(np.nanmean(inside)) if inside.size else float("nan")
    y_far = float(np.nanmean(far)) if far.size else float("nan")
    mid = y[(d > hull) & (d < np.quantile(d, 0.5))] if d.size >= 8 else y
    y_mid = float(np.nanmean(mid)) if mid.size else float("nan")
    cliff = bool(
        np.isfinite(y_in)
        and np.isfinite(y_mid)
        and y_in > _EPS
        and (y_in - y_mid) / y_in > 0.5
        and abs(y_mid - y_far) / max(y_in, _EPS) < 0.25
    )
    return {
        "r_norm_inside": y_in,
        "r_norm_far": y_far,
        "shape": "cliff" if cliff else "smooth_or_flat",
        "no_ground_truth": True,
        "claim": (
            "off-support the model reverts to its training mean; if E[R]≈0 that "
            "mean is near-1D. Residual-specific 1D prior needs a direct-TF contrast."
        ),
    }


def dummy_residual_mean_predictor(batch: dict[str, np.ndarray]) -> np.ndarray:
    """Placebo: R̂ = 0 (Haskell). Used when no ckpt; also the residual asymptote."""
    n = int(np.asarray(batch["vs1"]).shape[0])
    n_freq = int(batch.get("n_freq", 64))
    return np.zeros((n, n_freq), dtype=np.float64)


def represent_vs_use(
    latent: np.ndarray,
    target: np.ndarray,
    error: np.ndarray,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Ridge + MLP probes with a held-out split; injection = extra scalar in readout.

    'Not represented' requires *both* probes to fail. Causal 'used': inject the
    scalar as an extra decoder input with latent frozen; error drop ⇒ unused.
    """
    z = np.asarray(latent, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64).ravel()
    err = np.asarray(error, dtype=np.float64).ravel()
    n = z.shape[0]
    if n < 20:
        raise ValueError("need ≥20 samples for a probe split")
    idx = np.arange(n)
    tr, te = train_test_split(idx, test_size=0.3, random_state=seed)
    ridge = Ridge(alpha=1.0).fit(z[tr], y[tr])
    mlp = MLPRegressor(
        hidden_layer_sizes=(32,),
        max_iter=200,
        random_state=seed,
        alpha=1e-3,
    ).fit(z[tr], y[tr])
    r2_ridge = float(ridge.score(z[te], y[te]))
    r2_mlp = float(mlp.score(z[te], y[te]))
    yhat = ridge.predict(z[te])
    # Univariate channel–error is *not* the use test; injection is.
    z_inj_tr = np.column_stack([z[tr], y[tr][:, None]])
    z_inj_te = np.column_stack([z[te], y[te][:, None]])
    ridge_inj = Ridge(alpha=1.0).fit(z_inj_tr, err[tr])
    ridge_base = Ridge(alpha=1.0).fit(z[tr], err[tr])
    mse_base = float(np.mean((ridge_base.predict(z[te]) - err[te]) ** 2))
    mse_inj = float(np.mean((ridge_inj.predict(z_inj_te) - err[te]) ** 2))
    not_rep = (r2_ridge < 0.1) and (r2_mlp < 0.1)
    unused = (not not_rep) and (mse_inj < 0.9 * mse_base)
    used = (not not_rep) and (not unused)
    return {
        "r2_ridge": r2_ridge,
        "r2_mlp": r2_mlp,
        "injection_mse_drop": float(mse_base - mse_inj),
        "not_represented": bool(not_rep),
        "represented_but_unused": bool(unused),
        "represented_and_used": bool(used),
        "n_train": float(len(tr)),
        "n_test": float(len(te)),
        "probe_pred_te": yhat,
    }


def pack_dct_from_leftover(
    r_hat: np.ndarray,
    r: np.ndarray,
    *,
    cutoff: int = FNO_FREQ_MODES,
) -> dict[str, Any]:
    return dct_transfer_ratio(r_hat, r, cutoff=cutoff)


def extract_iid_branch_latents(
    *,
    ckpt_path: Path,
    cache_dir: Path,
    test_idx: np.ndarray,
    batch_size: int = 8,
    n_freq: int = 1000,
) -> dict[str, np.ndarray]:
    """Forward the ship ckpt on nested IID test; return pooled GNO/branch latents."""
    import torch
    from torch.utils.data import DataLoader

    from data import ResidualDeepONetDataset
    from eval_ood import _load_residual_model
    from model import gno_core
    from train import _device, _forward, apply_norms

    device = _device()
    model, blob, stats, trunk_set = _load_residual_model(ckpt_path, device)
    serial = bool(blob.get("serial_tf1d", True))
    ds = ResidualDeepONetDataset(
        cache_dir,
        test_idx,
        target="R_nom",
        trunk_set=trunk_set,
        n_freq=n_freq,
        serial_tf1d=serial,
    )
    apply_norms(ds, stats)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    mode = blob.get("branch_mode", "single")
    branches: list[np.ndarray] = []
    node_var: list[np.ndarray] = []
    core = gno_core(model)
    with torch.no_grad():
        for batch in loader:
            _ = _forward(
                model,
                batch["fields"].to(device),
                batch["stoch"].to(device),
                batch["trunk_y"].to(device),
                mode,
            )
            b = core.last_branch
            nodes = core.last_nodes
            if b is None or nodes is None:
                raise RuntimeError("GNO core did not stash last_branch / last_nodes")
            branches.append(b.mean(dim=1).cpu().numpy())
            node_var.append(nodes.var(dim=1).mean(dim=-1).cpu().numpy())
    return {
        "branch": np.concatenate(branches, axis=0),
        "node_spatial_var": np.concatenate(node_var, axis=0),
    }
