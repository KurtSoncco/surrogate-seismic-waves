"""Loaders for caches already on disk. No retraining."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

import config  # noqa: E402

from response_variability.gino_bias import (  # noqa: E402
    band_mask,
    leftover,
    trough_keep_mask,
)
from response_variability.metrics import central_recorder, pearson  # noqa: E402
from response_variability.plots.plot_presentation import nominal_vs_stairs  # noqa: E402

RESULTS = _EXP / "results"
PRESENTATION = RESULTS / "presentation"
CKPT = _EXP / "checkpoints" / "M7680_gino_rebal_ft.pt"
METRICS_JSON = RESULTS / "arch_train" / "M7680_gino_rebal_ft.json"
COMPARE_JSON = RESULTS / "compare_gino_loglo" / "summary.json"
RANK_CSV = RESULTS / "response_variability" / "eval_bias" / "ood_method_ranking.csv"
BAND_CSV = RESULTS / "response_variability" / "eval_bias" / "iid_band_misfit.csv"
SLOPE_JSON = (
    RESULTS
    / "response_variability"
    / "gino_bias"
    / "M7680_gino_rebal_ft"
    / "summary.json"
)
IID_HELDOUT = RESULTS / "response_variability" / "tf_atlas" / "iid_heldout_pack.npz"
DIP_HELDOUT = RESULTS / "response_variability" / "tf_atlas" / "dipping_heldout_pack.npz"

PACKS = {
    "iid": PRESENTATION / "iid_pack.npz",
    "dipping": PRESENTATION / "dipping_pack.npz",
    "three_layer": PRESENTATION / "three_layer_pack.npz",
}

# JSON domain key -> paper domain key.
_JSON_DOMAIN = {
    "iid": "iid",
    "ood_dipping": "dipping",
    "ood_three_layer": "three_layer",
}

# Ranking-CSV method name -> paper style name.
RANK_NAME = {
    "GINO": "GINO",
    "1D Base Case": "1D base case",
    "Pretell median": "Pretell median",
    "Pretell percentile": "Pretell percentile",
    "Toro Vs": "Toro",
    "Passeri tts": "Passeri",
    "Dmult": "Dmult",
}


def load_pack(domain: str) -> dict[str, np.ndarray]:
    path = PACKS[domain]
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def load_history() -> tuple[list[dict], int]:
    import torch

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    history = list(ckpt["history"])
    best = int(ckpt["best_epoch"])
    return history, best


def load_test_metrics() -> dict[str, dict[str, float]]:
    raw = json.loads(METRICS_JSON.read_text())
    out: dict[str, dict[str, float]] = {}
    for key, block in raw["test_by_domain"].items():
        out[_JSON_DOMAIN[key]] = {k: float(v) for k, v in block.items()}
    return out


def load_compare() -> dict[str, dict[str, float]]:
    """Relative L2 of |TF| for Haskell, LOGLO, and GINO on the nested tests."""
    raw = json.loads(COMPARE_JSON.read_text())
    models = raw["models"]
    out: dict[str, dict[str, float]] = {"haskell": {}, "loglo": {}, "gino": {}}
    for src, dst in (
        ("haskell_nom", "haskell"),
        ("loglo_pod_full7680", "loglo"),
        ("gino_rebal_ft", "gino"),
    ):
        for key, block in models[src].items():
            if key not in _JSON_DOMAIN or not isinstance(block, dict):
                continue
            out[dst][_JSON_DOMAIN[key]] = float(block["rel_l2_TF"])
    return out


def load_ranking() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with RANK_CSV.open() as f:
        for row in csv.DictReader(f):
            if row["method"] not in RANK_NAME:
                continue
            domain = {"iid": "iid", "dipping": "dipping", "three_layer": "three_layer"}[
                row["domain"]
            ]
            rows.append(
                {
                    "domain": domain,
                    "method": RANK_NAME[row["method"]],
                    "pearson": float(row["pearson_median"]),
                    "pearson_lo": float(row["pearson_lo"]),
                    "pearson_hi": float(row["pearson_hi"]),
                    "anderson": float(row["gof_median"]),
                    "anderson_lo": float(row["gof_lo"]),
                    "anderson_hi": float(row["gof_hi"]),
                }
            )
    return rows


def load_iid_bands() -> dict[str, dict[str, float]]:
    """Mean Pearson and Anderson by band. Columns match SUMMARY §5.1 (all recorders)."""
    import pandas as pd

    df = pd.read_csv(BAND_CSV)
    bands = ("low", "mid", "high")
    out: dict[str, dict[str, float]] = {}
    for src, name in RANK_NAME.items():
        sub = df[df["method"] == src]
        if sub.empty:
            continue
        block: dict[str, float] = {}
        for band in bands:
            block[f"pearson_{band}"] = float(sub[f"pearson_{band}"].mean())
            block[f"anderson_{band}"] = float(sub[f"gof_{band}"].mean())
        out[name] = block
    return out


def load_slopes() -> dict[str, dict[str, float]]:
    raw = json.loads(SLOPE_JSON.read_text())
    out: dict[str, dict[str, float]] = {}
    for domain in ("iid", "dipping", "three_layer"):
        s = raw["domains"][domain]["slope"]
        out[domain] = {
            "a": float(s["a"]),
            "b": float(s["b"]),
            "b_lo": float(s["b_lo"]),
            "b_hi": float(s["b_hi"]),
        }
    return out


def central_pearson_batch(tf_true: np.ndarray, tf_pred: np.ndarray) -> np.ndarray:
    """Pearson of |TF| on the central recorder, one value per sample."""
    n = int(tf_true.shape[0])
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = pearson(central_recorder(tf_true[i]), central_recorder(tf_pred[i]))
    return out


def val_test_pearson(domain: str) -> dict[str, np.ndarray]:
    path = IID_HELDOUT if domain == "iid" else DIP_HELDOUT
    with np.load(path, allow_pickle=True) as z:
        split = np.asarray(z["split"]).astype(str)
        scores = central_pearson_batch(z["tf_opensees"], z["tf_gino"])
    return {
        "val": scores[split == "val"],
        "test": scores[split == "test"],
    }


def nearest_index(values: np.ndarray, target: float, mask: np.ndarray | None = None) -> int:
    score = np.abs(np.asarray(values, dtype=float) - float(target))
    if mask is not None:
        score = np.where(np.asarray(mask, dtype=bool), score, np.inf)
    return int(np.argmin(score))


def pick_iid_cases(pack: dict[str, np.ndarray]) -> dict[str, int]:
    """Median-Pearson case, and a high-CoV case near the median of that subset."""
    pearson_g = np.asarray(pack["pearson_gino"], dtype=float)
    cov = np.asarray(pack["cov"], dtype=float)
    typical = nearest_index(pearson_g, float(np.nanmedian(pearson_g)))
    rough_cut = float(np.nanquantile(cov, 0.75))
    rough_mask = cov >= rough_cut
    rough_target = float(np.nanmedian(pearson_g[rough_mask]))
    rough = nearest_index(pearson_g, rough_target, rough_mask)
    return {"typical": typical, "rough": rough}


def pick_median_case(pack: dict[str, np.ndarray]) -> int:
    pearson_g = np.asarray(pack["pearson_gino"], dtype=float)
    return nearest_index(pearson_g, float(np.nanmedian(pearson_g)))


def pick_recovery_case(pack: dict[str, np.ndarray]) -> int:
    """Dipping case where GINO is at least typical and 1D is clearly behind."""
    g = np.asarray(pack["pearson_gino"], dtype=float)
    one = np.asarray(pack["pearson_1d"], dtype=float)
    eligible = g >= float(np.nanmedian(g))
    gap = np.where(eligible, g - one, -np.inf)
    return int(np.nanargmax(gap))


def z_max(pack: dict[str, np.ndarray], i: int) -> float:
    col = np.asarray(pack["vs_column"][i])
    finite = np.isfinite(col)
    nz = int(np.where(finite)[0][-1]) + 1 if bool(finite.any()) else 16
    h = float(pack["H"][i])
    nz = max(nz, int(np.ceil(h / float(config.DZ))) + 12, 16)
    return nz * float(config.DZ)


def site_section(pack: dict[str, np.ndarray], i: int) -> dict[str, np.ndarray | float]:
    """2-D Vs section down to the stored depth, with recorder positions in metres."""
    field = np.asarray(pack["vs_2d"][i], dtype=float)
    rows = np.where(np.isfinite(field).any(axis=1))[0]
    nz = int(rows[-1]) + 1 if rows.size else field.shape[0]
    return {
        "vs": field[:nz],
        "width": field.shape[1] * float(config.DX),
        "depth": nz * float(config.DZ),
        "recorder_x": np.asarray(pack["recorder_x"], dtype=float) * float(config.DX),
    }


def vs_profiles(pack: dict[str, np.ndarray], i: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Nominal 1-layer column, plus the true two-soil site when H1 and H2 exist."""
    depth = z_max(pack, i)
    nominal = nominal_vs_stairs(
        vs1=float(pack["vs1"][i]),
        H=float(pack["H"][i]),
        vs2=float(pack["vs2"][i]),
        z_max=depth,
    )
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {"nominal": nominal}
    h1 = float(pack["H1"][i]) if "H1" in pack else float("nan")
    h2 = float(pack["H2"][i]) if "H2" in pack else float("nan")
    vs_mid = float(pack["vs_mid"][i]) if "vs_mid" in pack else float("nan")
    if min(h1, h2) > 0 and np.isfinite(vs_mid):
        out["site"] = nominal_vs_stairs(
            vs1=float(pack["vs1"][i]),
            H=float(pack["H"][i]),
            vs2=float(pack["vs2"][i]),
            z_max=depth,
            h1=h1,
            h2=h2,
            vs_mid=vs_mid,
        )
    return out


def central_curves(pack: dict[str, np.ndarray], i: int) -> dict[str, np.ndarray]:
    ops = central_recorder(pack["tf_opensees"][i])
    nom = central_recorder(pack["tf_haskell_nominal"][i])
    gino = central_recorder(pack["tf_gino"][i])
    pretell = np.asarray(pack["tf_pretell"][i], dtype=np.float64)
    return {
        "freq": np.asarray(pack["freq"], dtype=float),
        "opensees": ops,
        "haskell": nom,
        "gino": gino,
        "pretell": pretell,
        "R": ops - nom,
        "Rhat": gino - nom,
    }


def calibration_cloud(
    pack: dict[str, np.ndarray], *, n_max: int = 6000, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Trough-windowed central-recorder pairs (R, Rhat), subsampled for the scatter."""
    r_hat = leftover(pack["tf_gino"], pack["tf_haskell_nominal"])
    r = leftover(pack["tf_opensees"], pack["tf_haskell_nominal"])
    rec = r.shape[1] // 2
    ops_c = np.asarray(pack["tf_opensees"], dtype=np.float64)[:, rec, :]
    keep = trough_keep_mask(ops_c) & band_mask(pack["freq"])[None, :]
    x = r[:, rec, :][keep].ravel()
    y = r_hat[:, rec, :][keep].ravel()
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size > n_max:
        rng = np.random.default_rng(seed)
        take = rng.choice(x.size, n_max, replace=False)
        x, y = x[take], y[take]
    return x, y


def case_blurb(pack: dict[str, np.ndarray], i: int) -> str:
    parts = [
        f"Pearson {float(pack['pearson_gino'][i]):.2f}",
        f"CoV {float(pack['cov'][i]):.2f}",
        f"H {float(pack['H'][i]):.0f} m",
        f"Vs1 {float(pack['vs1'][i]):.0f}",
    ]
    h1 = float(pack["H1"][i]) if "H1" in pack else float("nan")
    h2 = float(pack["H2"][i]) if "H2" in pack else float("nan")
    if np.isfinite(h1) and h1 > 0 and np.isfinite(h2) and h2 > 0:
        parts.append(f"H1 {h1:.0f} m")
        parts.append(f"H2 {h2:.0f} m")
    return ", ".join(parts)
