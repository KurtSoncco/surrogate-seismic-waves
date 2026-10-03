#!/usr/bin/env python3
"""Dipping val+test: uniform interface depth for Toro, then Passeri.

Depth is uniform along the dip and independent of the case Vs2. Toro keeps its
lognormal soil Vs; Passeri keeps its lognormal travel-time Vs. The published
band is the equal-weight min/max across depths, not exp(±σ_ln).

    uv run python experiments/DeepONet-Residual/response_variability/evals/eval_dip_uniform.py
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_EXP = Path(__file__).resolve().parents[2]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

import config  # noqa: E402

from response_variability.dip_depth import (  # noqa: E402
    DIP_SPAN_M,
    N_DEPTHS,
    N_STRIP,
    depth_range,
    empirical_sigma_y,
    sigma_y,
    uniform_dip_depths,
    uniform_spectrum,
)
from response_variability.names import (  # noqa: E402
    DMULT,
    GINO,
    HASKELL_NOMINAL,
    PASSERI,
    PRETELL,
    TORO,
)
from response_variability.style import apply_nature_style, panel_letter, savefig  # noqa: E402
from response_variability.plots.tf_atlas import (  # noqa: E402
    ATLAS_CURVE_STYLE,
    OUT_DIR,
    atlas_panel_title,
    plot_atlas_pages,
)

HELD_OUT = OUT_DIR / "dipping_heldout_pack.npz"
VAL_PACK = OUT_DIR / "dipping_val_pack.npz"
TEST_PACK = config.RESULTS_DIR / "presentation" / "dipping_pack.npz"
CLASSICAL = (
    config.RESULTS_DIR / "response_variability" / "eval_bias" / "dipping_classical.npz"
)
PACK_OUT = OUT_DIR / "dipping_uniform_pack.npz"
CHECKPOINT = OUT_DIR / "dipping_uniform_checkpoint.npz"
FIG_DIR = OUT_DIR / "dipping"

_FREQ: np.ndarray | None = None


def _init_worker(freq: np.ndarray) -> None:
    global _FREQ
    _FREQ = np.asarray(freq, dtype=np.float64)


def _case_task(task: dict) -> dict:
    """Toro and Passeri at the same uniform depths, via seiskit. Vs2 stays the case value."""
    from response_variability.seiskit_arms import hallal_geomean_tf

    if _FREQ is None:
        raise RuntimeError("worker frequency grid was not initialized")
    depths = uniform_dip_depths(
        float(task["H"]),
        float(task["theta"]),
        L_h=float(task["L_h"]),
        n=int(task["n_depths"]),
    )
    kinds = tuple(task.get("kinds", ("toro", "passeri")))
    out: dict = {"i": int(task["i"])}
    for kind in kinds:
        rows = []
        for depth in depths:
            geo, _sig = hallal_geomean_tf(
                freq=_FREQ,
                vs1=float(task["vs1"]),
                H=float(depth),
                cov=float(task["cov"]),
                vs2=float(task["vs2"]),
                xi=float(task["xi"]),
                n_seeds=int(task["n_seeds"]),
                kind=kind,
            )
            rows.append(geo)
        geo, lo, hi = uniform_spectrum(np.vstack(rows))
        out[f"{kind}_geo"] = geo
        out[f"{kind}_lo"] = lo
        out[f"{kind}_hi"] = hi
    return out


def _split_fields(held: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    """Val then test Vs strips, aligned with ``held`` (val concatenated with test)."""
    val = np.load(VAL_PACK, allow_pickle=True)
    test = np.load(TEST_PACK, allow_pickle=True)
    if not np.array_equal(
        held["sample_idx"][: val["sample_idx"].shape[0]], val["sample_idx"]
    ):
        raise RuntimeError("val pack sample_idx does not match the held-out prefix")
    n_val = int(val["sample_idx"].shape[0])
    if not np.array_equal(held["sample_idx"][n_val:], test["sample_idx"]):
        raise RuntimeError("test pack sample_idx does not match the held-out suffix")
    return np.asarray(val["vs_2d"]), np.asarray(test["vs_2d"])


def xi_for_heldout(held: np.lib.npyio.NpzFile) -> np.ndarray:
    """Match the xi used for the frozen-H spectra.

    Val Toro/Passeri were built without a per-case xi, so they use the Hallal
    default. Test spectra come from the classical pack, which stores xi_mean.
    """
    n = int(held["tf_opensees"].shape[0])
    xi = np.full(n, float(config.DEFAULT_XI_TREND), dtype=np.float64)
    split = np.asarray(held["split"])
    classical = np.load(CLASSICAL, allow_pickle=True)
    test = split == "test"
    if int(np.sum(test)) != int(classical["xi_mean"].shape[0]):
        raise RuntimeError("classical xi_mean length does not match the test split")
    if not np.array_equal(held["sample_idx"][test], classical["sample_idx"]):
        raise RuntimeError("classical sample_idx does not match the test split")
    xi[test] = np.asarray(classical["xi_mean"], dtype=np.float64)
    return xi


def depth_table(held: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    """Analytical vs empirical interface std, one row per val and test case."""
    val_vs, test_vs = _split_fields(held)
    n_val = int(val_vs.shape[0])
    n = int(held["H"].shape[0])
    if n_val + int(test_vs.shape[0]) != n:
        raise RuntimeError("val and test strips do not cover the held-out pack")
    split = np.asarray(held["split"])
    angle = np.asarray(held["dip_angle_deg"], dtype=np.float64)
    H = np.asarray(held["H"], dtype=np.float64)
    vs2 = np.asarray(held["vs2"], dtype=np.float64)
    emp = np.empty(n, dtype=np.float64)
    disc = np.empty(n, dtype=np.float64)
    cont = np.empty(n, dtype=np.float64)
    span = np.empty(n, dtype=np.float64)
    for i in range(n):
        field = val_vs[i] if split[i] == "val" else test_vs[i - n_val]
        emp[i] = empirical_sigma_y(field, float(vs2[i]), dz=float(config.DZ))
        disc[i] = sigma_y(DIP_SPAN_M, float(angle[i]), n=N_STRIP)
        cont[i] = sigma_y(DIP_SPAN_M, float(angle[i]), n=None)
        span[i] = depth_range(DIP_SPAN_M, float(angle[i]))
    return {
        "split": split.astype(str),
        "sample": np.arange(n, dtype=int),
        "local_idx": np.asarray(held["local_idx"], dtype=int),
        "sample_idx": np.asarray(held["sample_idx"], dtype=int),
        "dip_angle_deg": angle,
        "H": H,
        "vs2": vs2,
        "sigma_y_empirical": emp,
        "sigma_y_discrete": disc,
        "sigma_y_continuous": cont,
        "depth_range_m": span,
        "n_strip": np.full(n, N_STRIP, dtype=int),
    }


def write_depth_tables(table: dict[str, np.ndarray], fig_dir: Path) -> None:
    import pandas as pd

    fig_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(table)
    frame["abs_err_discrete"] = np.abs(
        frame["sigma_y_empirical"] - frame["sigma_y_discrete"]
    )
    frame["abs_err_continuous"] = np.abs(
        frame["sigma_y_empirical"] - frame["sigma_y_continuous"]
    )
    frame.to_csv(fig_dir / "depth_sigma.csv", index=False)
    rows = []
    for split, part in frame.groupby("split", sort=False):
        rows.append(
            {
                "split": split,
                "n": int(len(part)),
                "median_abs_err_discrete_m": float(part["abs_err_discrete"].median()),
                "median_abs_err_continuous_m": float(
                    part["abs_err_continuous"].median()
                ),
                "max_abs_err_discrete_m": float(part["abs_err_discrete"].max()),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(fig_dir / "depth_sigma_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)

    import matplotlib.pyplot as plt

    apply_nature_style()
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    colors = {"val": "#0072B2", "test": "#D55E00"}
    for split, part in frame.groupby("split", sort=False):
        ax.scatter(
            part["sigma_y_discrete"],
            part["sigma_y_empirical"],
            s=18,
            c=colors.get(str(split), "#333333"),
            label=str(split),
            alpha=0.85,
            linewidths=0,
        )
    lim = float(
        np.nanmax(
            np.concatenate([frame["sigma_y_discrete"], frame["sigma_y_empirical"]])
        )
    )
    hi = max(lim * 1.05, 0.1)
    ax.plot([0, hi], [0, hi], color="0.4", lw=0.8, zorder=0)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"discrete $\sigma_y$ (m)")
    ax.set_ylabel(r"empirical interface std (m)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title(r"Uniform dip depth, $N=500$")
    savefig(fig, fig_dir / "depth_sigma_scatter.png")


def _empty_spectra(n: int, n_freq: int) -> dict[str, np.ndarray]:
    shape = (n, n_freq)
    return {
        "toro_geo": np.full(shape, np.nan, dtype=np.float64),
        "toro_lo": np.full(shape, np.nan, dtype=np.float64),
        "toro_hi": np.full(shape, np.nan, dtype=np.float64),
        "passeri_geo": np.full(shape, np.nan, dtype=np.float64),
        "passeri_lo": np.full(shape, np.nan, dtype=np.float64),
        "passeri_hi": np.full(shape, np.nan, dtype=np.float64),
        "done": np.zeros(n, dtype=bool),
    }


def _load_checkpoint(n: int, n_freq: int) -> dict[str, np.ndarray]:
    if not CHECKPOINT.is_file():
        return _empty_spectra(n, n_freq)
    z = np.load(CHECKPOINT)
    if int(z["done"].shape[0]) != n or int(z["toro_geo"].shape[1]) != n_freq:
        return _empty_spectra(n, n_freq)
    return {
        key: np.array(z[key])
        for key in (
            "toro_geo",
            "toro_lo",
            "toro_hi",
            "passeri_geo",
            "passeri_lo",
            "passeri_hi",
            "done",
        )
    }


def _save_checkpoint(spec: dict[str, np.ndarray]) -> None:
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CHECKPOINT, **spec)


def run_ensembles(
    held: np.lib.npyio.NpzFile,
    *,
    n_depths: int,
    n_seeds: int,
    workers: int,
) -> dict[str, np.ndarray]:
    freq = np.asarray(held["freq"], dtype=np.float64)
    n = int(held["H"].shape[0])
    spec = _load_checkpoint(n, int(freq.shape[0]))
    xi = xi_for_heldout(held)
    kinds = ("toro", "passeri")
    pending = []
    for i in range(n):
        if spec["done"][i]:
            continue
        pending.append(
            {
                "i": i,
                "vs1": float(held["vs1"][i]),
                "H": float(held["H"][i]),
                "cov": float(held["cov"][i]),
                "vs2": float(held["vs2"][i]),
                "xi": float(xi[i]),
                "theta": float(held["dip_angle_deg"][i]),
                "L_h": DIP_SPAN_M,
                "n_depths": int(n_depths),
                "n_seeds": int(n_seeds),
                "kinds": kinds,
            }
        )
    print(
        f"Uniform-depth ensembles: {len(pending)} cases pending, "
        f"{n_depths} depths, {n_seeds} seeds, {workers} workers",
        flush=True,
    )
    if not pending:
        return spec
    t0 = time.perf_counter()
    finished = 0
    with ProcessPoolExecutor(
        max_workers=int(workers),
        mp_context=__import__("multiprocessing").get_context("fork"),
        initializer=_init_worker,
        initargs=(freq,),
    ) as pool:
        futures = {pool.submit(_case_task, task): task["i"] for task in pending}
        for fut in as_completed(futures):
            result = fut.result()
            i = int(result["i"])
            for kind in kinds:
                spec[f"{kind}_geo"][i] = result[f"{kind}_geo"]
                spec[f"{kind}_lo"][i] = result[f"{kind}_lo"]
                spec[f"{kind}_hi"][i] = result[f"{kind}_hi"]
            if set(kinds) >= {"toro", "passeri"}:
                spec["done"][i] = True
            finished += 1
            if finished % 8 == 0 or finished == len(pending):
                _save_checkpoint(spec)
                elapsed = time.perf_counter() - t0
                rate = finished / max(elapsed, 1e-6)
                left = (len(pending) - finished) / max(rate, 1e-6)
                print(
                    f"  {int(spec['done'].sum())}/{n} done "
                    f"({elapsed / 60:.1f} min elapsed, ~{left / 60:.1f} min left)",
                    flush=True,
                )
    if not bool(np.all(spec["done"])):
        raise RuntimeError("ensemble checkpoint is incomplete")
    return spec


def build_pack(
    held: np.lib.npyio.NpzFile, spec: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    drop = {"vs_2d", "vs_column"}
    pack = {key: held[key] for key in held.files if key not in drop}
    pack["tf_toro_frozen"] = np.array(held["tf_toro"], dtype=np.float64, copy=True)
    pack["tf_passeri_frozen"] = np.array(
        held["tf_passeri"], dtype=np.float64, copy=True
    )
    pack["sigma_ln_toro_frozen"] = np.array(
        held["sigma_ln_toro"], dtype=np.float64, copy=True
    )
    pack["sigma_ln_passeri_frozen"] = np.array(
        held["sigma_ln_passeri"], dtype=np.float64, copy=True
    )
    pack["tf_toro"] = spec["toro_geo"]
    pack["tf_toro_lo"] = spec["toro_lo"]
    pack["tf_toro_hi"] = spec["toro_hi"]
    pack["tf_passeri"] = spec["passeri_geo"]
    pack["tf_passeri_lo"] = spec["passeri_lo"]
    pack["tf_passeri_hi"] = spec["passeri_hi"]
    # The published band is the uniform envelope, not a lognormal σ_ln.
    pack["sigma_ln_toro"] = np.full_like(spec["toro_geo"], np.nan)
    pack["sigma_ln_passeri"] = np.full_like(spec["passeri_geo"], np.nan)
    return pack


def _center(tf: np.ndarray, i: int) -> np.ndarray:
    a = np.asarray(tf[i], dtype=np.float64)
    if a.ndim == 1:
        return a
    return a[a.shape[0] // 2]


def write_pearson(pack: dict[str, np.ndarray], fig_dir: Path) -> None:
    import pandas as pd

    from response_variability.metrics import band_pearson

    methods = (
        (GINO, "tf_gino"),
        (HASKELL_NOMINAL, "tf_haskell_nominal"),
        (TORO, "tf_toro"),
        ("Toro frozen-H", "tf_toro_frozen"),
        (PASSERI, "tf_passeri"),
        ("Passeri frozen-H", "tf_passeri_frozen"),
        (DMULT, "tf_dmult"),
        (PRETELL, "tf_pretell"),
    )
    freq = pack["freq"]
    split = np.asarray(pack["split"]).astype(str)
    rows = []
    for name, key in methods:
        if key not in pack:
            continue
        for part in ("val", "test"):
            idx = np.flatnonzero(split == part)
            scores = np.array(
                [
                    band_pearson(
                        _center(pack[key], int(i)),
                        _center(pack["tf_opensees"], int(i)),
                        freq,
                        lo=0.1,
                        hi=10.0,
                    )
                    for i in idx
                ],
                dtype=np.float64,
            )
            rows.append(
                {
                    "split": part,
                    "method": name,
                    "n": int(scores.size),
                    "median": float(np.nanmedian(scores)),
                    "p25": float(np.nanpercentile(scores, 25)),
                    "p75": float(np.nanpercentile(scores, 75)),
                }
            )
    frame = pd.DataFrame(rows)
    fig_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(fig_dir / "pearson_by_split.csv", index=False)
    print(frame.to_string(index=False), flush=True)


def _pick_visible(pack: dict[str, np.ndarray], split: str) -> int:
    mask = np.asarray(pack["split"]).astype(str) == split
    angle = np.abs(np.asarray(pack["dip_angle_deg"], dtype=np.float64))
    angle = np.where(mask, angle, -np.inf)
    return int(np.argmax(angle))


def plot_bands(pack: dict[str, np.ndarray], path: Path) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    apply_nature_style()
    freq = np.asarray(pack["freq"], dtype=np.float64)
    toro_c = ATLAS_CURVE_STYLE[TORO]["color"]
    pas_c = ATLAS_CURVE_STYLE[PASSERI]["color"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), sharey=True)
    for ax, split, letter in zip(axes, ("val", "test"), "ab"):
        i = _pick_visible(pack, split)
        ops = np.asarray(pack["tf_opensees"][i], dtype=np.float64)
        for r in range(ops.shape[0]):
            ax.plot(freq, np.maximum(ops[r], 1e-6), color="0.75", lw=0.7, zorder=1)
        for frozen_key, sig_key, color in (
            ("tf_toro_frozen", "sigma_ln_toro_frozen", toro_c),
            ("tf_passeri_frozen", "sigma_ln_passeri_frozen", pas_c),
        ):
            geo = np.asarray(pack[frozen_key][i], dtype=np.float64)
            sig = np.asarray(pack[sig_key][i], dtype=np.float64)
            lo = np.maximum(geo * np.exp(-sig), 1e-6)
            hi = np.maximum(geo * np.exp(sig), 1e-6)
            ax.fill_between(freq, lo, hi, color=color, alpha=0.12, lw=0, zorder=2)
            ax.plot(
                freq,
                np.maximum(geo, 1e-6),
                color=color,
                ls=(0, (1.2, 1.4)),
                lw=1.1,
                zorder=3,
            )
        for lo_key, hi_key, geo_key, color in (
            ("tf_toro_lo", "tf_toro_hi", "tf_toro", toro_c),
            ("tf_passeri_lo", "tf_passeri_hi", "tf_passeri", pas_c),
        ):
            lo = np.maximum(np.asarray(pack[lo_key][i], dtype=np.float64), 1e-6)
            hi = np.maximum(np.asarray(pack[hi_key][i], dtype=np.float64), 1e-6)
            geo = np.maximum(np.asarray(pack[geo_key][i], dtype=np.float64), 1e-6)
            ax.fill_between(freq, lo, hi, color=color, alpha=0.28, lw=0, zorder=4)
            ax.plot(freq, geo, color=color, lw=1.8, zorder=5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.1, 10.0)
        ax.set_ylim(1e-1, 5e1)
        ax.set_xlabel(r"$f$ (Hz)")
        ax.set_title(atlas_panel_title(pack, i), fontsize=8)
        panel_letter(ax, letter, x=0.02, y=0.96)
    axes[0].set_ylabel(r"$|\mathrm{TF}|$")
    handles = [
        Line2D([0], [0], color="0.75", lw=1.2, label="2D stations"),
        Patch(facecolor=toro_c, alpha=0.12, label=r"Toro frozen $\pm 1\sigma_{\ln}$"),
        Patch(facecolor=pas_c, alpha=0.12, label=r"Passeri frozen $\pm 1\sigma_{\ln}$"),
        Patch(facecolor=toro_c, alpha=0.28, label="Toro uniform depth"),
        Patch(facecolor=pas_c, alpha=0.28, label="Passeri uniform depth"),
        Line2D([0], [0], color=toro_c, lw=1.8, label="Toro geomean"),
        Line2D([0], [0], color=pas_c, lw=1.8, label="Passeri geomean"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return savefig(fig, path)


def save_pack(pack: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **pack)
    print(f"Wrote {path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-depths", type=int, default=N_DEPTHS)
    p.add_argument("--n-seeds", type=int, default=40)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--skip-ensembles", action="store_true")
    p.add_argument("--skip-plots", action="store_true")
    args = p.parse_args()

    held = np.load(HELD_OUT, allow_pickle=True)
    table = depth_table(held)
    write_depth_tables(table, FIG_DIR)

    if args.skip_ensembles and PACK_OUT.is_file():
        pack = dict(np.load(PACK_OUT, allow_pickle=True))
    else:
        spec = run_ensembles(
            held,
            n_depths=args.n_depths,
            n_seeds=args.n_seeds,
            workers=args.workers,
        )
        pack = build_pack(held, spec)
        save_pack(pack, PACK_OUT)

    if args.skip_plots:
        return
    write_pearson(pack, FIG_DIR)
    plot_bands(pack, FIG_DIR / "uniform_bands_vs_2d.png")
    pages = plot_atlas_pages(pack, OUT_DIR, "dipping")
    print(f"Atlas pages: {len(pages)}", flush=True)


if __name__ == "__main__":
    main()
