#!/usr/bin/env python3
"""Toro 2022 Vs randomization demo: which variables drive it, and what it does to the TF.

Draws an ensemble of frozen-H AR(1) Vs(z) realizations for one real IID case (same
call seiskit_arms.hallal_geomean_tf makes with ``kind="toro"``: hallal_config ->
generate_vs_randomized_profile, one draw per seed), runs each through the
Thomson-Haskell 1-D solver, and makes two figures:

  toro_vs_tf_demo.png
      Two panels: randomized Vs(z) profiles (median +/- 1 log-std) and the
      resulting |TF(f)| ensemble (geomean +/- 1 log-std), for the same case.

  toro_params_demo.png
      What actually drives the randomization for that case: the SPID sigma_ln
      V(z) depth taper, the Toro adjacent-layer correlation rho(z), the ln(Vs)
      marginal at the surface sample vs its clipped-normal target, and a table
      of every ``ProfileRandomizationConfig`` field the draw touches.

Usage: python -m response_variability.plots.plot_toro_demo [--case-index N] [--n-seeds N]
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

_EXP = Path(__file__).resolve().parents[2]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

import config  # noqa: E402

from haskell_baseline import haskell_af_within  # noqa: E402
from response_variability.metrics import spatial_sigma_ln  # noqa: E402
from response_variability.names import HASKELL_NOMINAL, METHOD_COLORS, TORO  # noqa: E402
from response_variability.plots.plot_presentation import load_pack, pack_path  # noqa: E402
from response_variability.seiskit_arms import ensure_seiskit, hallal_config  # noqa: E402
from response_variability.style import (  # noqa: E402
    apply_nature_style,
    figsize,
    panel_letter,
    savefig,
)

OUT_DIR = config.RESULTS_DIR / "presentation"
DZ = 0.5
LN_STD_Z = 1.0  # +/- 1 log-std envelope
BEDROCK_VIEW_M = 20.0  # visualization only -- vary_bedrock_vs=False, so it never affects the TF


def pick_case(pack: dict[str, np.ndarray], index: int | None) -> dict[str, float | int]:
    """A real case from the IID pack: median-H by default (a representative column)."""
    h = np.asarray(pack["H"], dtype=float)
    if index is None:
        index = int(np.argsort(h)[len(h) // 2])
    xi = (
        float(pack["xi_mean"][index])
        if "xi_mean" in pack and np.isfinite(pack["xi_mean"][index])
        else config.DEFAULT_XI_TREND
    )
    return {
        "index": index,
        "vs1": float(pack["vs1"][index]),
        "H": float(pack["H"][index]),
        "cov": float(pack["cov"][index]),
        "vs2": float(pack["vs2"][index]),
        "xi": xi if xi > 0 else config.DEFAULT_XI_TREND,
    }


def draw_ensemble(
    freq: np.ndarray,
    case: dict[str, float | int],
    *,
    n_seeds: int,
    dz: float = DZ,
    bedrock_view_m: float = BEDROCK_VIEW_M,
):
    """Return (cfg, depth_mid, vs_rows[n_seeds, n_soil+n_bedrock], tf_rows[n_seeds, n_freq], soil_nz).

    ``vs_rows`` keeps the bedrock samples ``generate_vs_randomized_profile`` appends below
    ``H`` (Hallal's ``vary_bedrock_vs=False`` forces them to the fixed ``vs2``, so that part
    of the band collapses to zero width -- that's the model, not a bug). ``bedrock_view_m``
    only widens how much of that fixed half-space is drawn; it is not read by
    ``haskell_af_within`` (which is capped at ``soil_nz`` and takes ``vs2`` separately), so it
    cannot change the transfer function.
    """
    ensure_seiskit()
    from seiskit.profile_randomization import generate_vs_randomized_profile

    cfg = hallal_config(
        vs1=case["vs1"], H=case["H"], cov=case["cov"], vs2=case["vs2"], dz=dz
    )
    cfg = dataclasses.replace(cfg, bedrock_thickness=bedrock_view_m)
    soil_nz = max(1, int(round(case["H"] / dz)))
    n_bedrock = max(1, int(round(bedrock_view_m / dz)))
    n_total = soil_nz + n_bedrock
    vs_rows = np.empty((n_seeds, n_total), dtype=np.float64)
    tf_rows = np.empty((n_seeds, freq.shape[0]), dtype=np.float64)
    for k, seed in enumerate(range(1, n_seeds + 1)):
        rng = np.random.default_rng(seed)
        vs = np.asarray(generate_vs_randomized_profile(cfg, rng), dtype=float)
        zeta = np.full_like(vs, float(case["xi"]))
        tf_rows[k] = haskell_af_within(
            freq, vs, zeta, dz=dz, vs_rock=case["vs2"], soil_nz=soil_nz
        )
        vs_rows[k] = vs[:n_total]
    depth_mid = (np.arange(n_total) + 0.5) * dz
    return cfg, depth_mid, vs_rows, tf_rows, soil_nz


def plot_vs_and_tf(
    freq: np.ndarray,
    case: dict[str, float | int],
    depth_mid: np.ndarray,
    vs_rows: np.ndarray,
    tf_rows: np.ndarray,
    soil_nz: int,
    dest: Path,
    *,
    n_show: int = 25,
) -> None:
    import matplotlib.pyplot as plt

    apply_nature_style()
    color = METHOD_COLORS[TORO]
    base_color = METHOD_COLORS[HASKELL_NOMINAL]

    ln_vs = np.log(vs_rows)
    vs_median = np.exp(np.median(ln_vs, axis=0))
    vs_sigma_ln = np.std(ln_vs, axis=0, ddof=1)
    vs_lo = vs_median * np.exp(-LN_STD_Z * vs_sigma_ln)
    vs_hi = vs_median * np.exp(LN_STD_Z * vs_sigma_ln)

    tf_geomean = np.exp(np.mean(np.log(np.clip(tf_rows, 1e-12, None)), axis=0))
    tf_sigma_ln = spatial_sigma_ln(tf_rows)
    tf_lo = tf_geomean * np.exp(-LN_STD_Z * tf_sigma_ln)
    tf_hi = tf_geomean * np.exp(LN_STD_Z * tf_sigma_ln)

    nominal_tf = haskell_af_within(
        freq,
        np.full(soil_nz, case["vs1"]),
        np.full(soil_nz, case["xi"]),
        dz=DZ,
        vs_rock=case["vs2"],
        soil_nz=soil_nz,
    )

    fig, (ax_vs, ax_tf) = plt.subplots(1, 2, figsize=figsize("double", height_mm=70))

    rng_show = np.random.default_rng(0)
    show_idx = rng_show.choice(vs_rows.shape[0], size=min(n_show, vs_rows.shape[0]), replace=False)
    for i in show_idx:
        ax_vs.step(vs_rows[i], depth_mid, color=color, alpha=0.15, lw=0.6, where="mid")
    ax_vs.fill_betweenx(depth_mid, vs_lo, vs_hi, color=color, alpha=0.25, lw=0, label=r"median $\pm\,1\sigma_{\ln}$")
    ax_vs.step(vs_median, depth_mid, color=color, lw=1.4, where="mid", label="median (randomized)")
    nominal_vs = np.where(depth_mid <= case["H"], case["vs1"], case["vs2"])
    ax_vs.step(
        nominal_vs, depth_mid, color=base_color, ls="-.", lw=1.2, where="mid",
        label=f"{HASKELL_NOMINAL} ($V_{{s1}}$/$V_{{s2}}$)",
    )
    ax_vs.axhline(case["H"], color="0.6", ls=":", lw=0.7)
    ax_vs.annotate(
        "soil / bedrock", (0.02, case["H"]), xycoords=("axes fraction", "data"),
        textcoords="offset points", xytext=(0, 3), fontsize=5.5, color="0.4",
    )
    # Zoom on the soil range -- Vs2 is ~5x Vs1 here, so a shared linear axis would
    # crush the soil variability that is the actual story. The nominal step still
    # runs to Vs2; it just leaves the visible window, which the annotation covers.
    soil_lo = float(np.min(vs_lo[:soil_nz]))
    soil_hi = float(np.max(vs_hi[:soil_nz]))
    pad = 0.12 * (soil_hi - soil_lo)
    ax_vs.set_xlim(soil_lo - pad, soil_hi + pad)
    ax_vs.annotate(
        f"bedrock $V_{{s2}}$={case['vs2']:.0f} m/s →",
        (0.98, case["H"] + 0.06 * (depth_mid[-1] - depth_mid[0])),
        xycoords=("axes fraction", "data"), ha="right", fontsize=5.5, color=base_color,
    )
    ax_vs.set_ylim(depth_mid[-1] + DZ, 0.0)
    ax_vs.set_xlabel(r"$V_s$ (m/s)")
    ax_vs.set_ylabel("Depth (m)")
    ax_vs.set_title(f"Randomized $V_s(z)$, n={vs_rows.shape[0]} seeds")
    ax_vs.legend(loc="lower left", fontsize=6)
    panel_letter(ax_vs, "a")

    for i in show_idx:
        ax_tf.loglog(freq, np.maximum(tf_rows[i], 1e-6), color=color, alpha=0.12, lw=0.5)
    ax_tf.fill_between(
        freq, np.maximum(tf_lo, 1e-6), np.maximum(tf_hi, 1e-6),
        color=color, alpha=0.25, lw=0, label=r"median $\pm\,1\sigma_{\ln}$",
    )
    ax_tf.loglog(freq, np.maximum(tf_geomean, 1e-6), color=color, lw=1.4, label=f"{TORO} geomean")
    ax_tf.loglog(
        freq, np.maximum(nominal_tf, 1e-6), color=base_color, ls="-.", lw=1.2, label=HASKELL_NOMINAL
    )
    ax_tf.set_xlabel("Frequency (Hz)")
    ax_tf.set_ylabel(r"$|\mathrm{TF}|$")
    ax_tf.set_title("Transfer function ensemble")
    ax_tf.legend(loc="lower left", fontsize=6)
    panel_letter(ax_tf, "b")

    fig.suptitle(
        f"Toro 2022 frozen-H AR(1) randomization -- case {case['index']}: "
        f"$V_{{s1}}$={case['vs1']:.0f} m/s, $H$={case['H']:.0f} m, "
        f"$V_{{s2}}$={case['vs2']:.0f} m/s",
        fontsize=8,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    savefig(fig, dest)
    plt.close(fig)


def plot_params(
    case: dict[str, float | int],
    cfg,
    depth_mid: np.ndarray,
    vs_rows: np.ndarray,
    soil_nz: int,
    n_seeds: int,
    dest: Path,
) -> None:
    import matplotlib.pyplot as plt

    from seiskit.profile_randomization import toro_adjacent_correlation, toro_sigma_ln_vs

    apply_nature_style()
    color = METHOD_COLORS[TORO]

    # Sigma taper / AR(1) rho only describe the randomized soil column; the bedrock
    # sample below H is fixed (vary_bedrock_vs=False), so it is excluded here.
    depth_mid = depth_mid[:soil_nz]
    vs_rows = vs_rows[:, :soil_nz]

    z_grid = np.linspace(0.0, depth_mid[-1], 200)
    sigma_z = toro_sigma_ln_vs(z_grid, cfg)
    rho_z = toro_adjacent_correlation(
        depth_mid,
        rho_0=cfg.toro_rho_0,
        delta=cfg.toro_delta,
        rho_200=cfg.toro_rho_200,
        b=cfg.toro_b,
        h0=cfg.toro_h0,
    )

    # Surface-sample ln(Vs) marginal vs its clipped-normal target.
    sigma_surf = float(toro_sigma_ln_vs(np.array([depth_mid[0]]), cfg).reshape(()))
    sigma_eff = cfg.toro_sigma_inflate * sigma_surf
    z_scores = (np.log(vs_rows[:, 0]) - np.log(case["vs1"])) / sigma_eff

    fig, axes = plt.subplots(
        2, 2, figsize=figsize("double", height_mm=140),
        gridspec_kw={"hspace": 0.65, "wspace": 0.3},
    )
    ax_sig, ax_rho, ax_hist, ax_tab = axes.ravel()

    ax_sig.plot(sigma_z, z_grid, color=color, lw=1.4)
    ax_sig.axhline(cfg.sigma_ln_vs_depth_m, color="0.5", ls=":", lw=0.8)
    ax_sig.scatter(
        [cfg.sigma_ln_vs_surface, cfg.sigma_ln_vs], [0.0, cfg.sigma_ln_vs_depth_m],
        color=color, s=14, zorder=5,
    )
    ax_sig.annotate(
        f"surface {cfg.sigma_ln_vs_surface:.2f}", (cfg.sigma_ln_vs_surface, 0.0),
        textcoords="offset points", xytext=(4, -8), fontsize=6,
    )
    ax_sig.annotate(
        f"{cfg.sigma_ln_vs_depth_m:.0f} m: {cfg.sigma_ln_vs:.2f}",
        (cfg.sigma_ln_vs, cfg.sigma_ln_vs_depth_m),
        textcoords="offset points", xytext=(4, 4), fontsize=6,
    )
    ax_sig.set_ylim(depth_mid[-1], 0.0)
    ax_sig.set_xlabel(r"SPID $\sigma_{\ln V}(z)$")
    ax_sig.set_ylabel("Depth (m)")
    ax_sig.set_title("Depth taper (before 1.16x inflation)")
    panel_letter(ax_sig, "a")

    ax_rho.plot(0.5 * (depth_mid[:-1] + depth_mid[1:]), rho_z, color=color, lw=1.4, marker="o", ms=2)
    ax_rho.set_xlabel("Depth of adjacent pair (m)")
    ax_rho.set_ylabel(r"Adjacent-layer $\rho$")
    ax_rho.set_ylim(0.0, 1.02)
    ax_rho.set_title(r"AR(1) correlation: $\rho_0$ e$^{-\Delta z/\delta}$ blended to $\rho_{200}$")
    panel_letter(ax_rho, "b")

    bins = np.linspace(-3.0, 3.0, 25)
    ax_hist.hist(z_scores, bins=bins, density=True, color=color, alpha=0.55, label="drawn $Z$ (surface)")
    zz = np.linspace(-3.0, 3.0, 400)
    ax_hist.plot(zz, np.exp(-0.5 * zz**2) / np.sqrt(2 * np.pi), color="0.2", lw=1.0, label=r"$N(0,1)$")
    ax_hist.axvline(cfg.clip_std, color="0.3", ls="--", lw=0.8)
    ax_hist.axvline(-cfg.clip_std, color="0.3", ls="--", lw=0.8, label=f"clip $|Z|\\leq${cfg.clip_std:g}")
    ax_hist.set_xlabel(r"$Z = \ln(V_s/V_{s1}) / (1.16\,\sigma_{\ln V})$ at surface")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title(f"Surface-sample marginal, n={n_seeds} seeds")
    ax_hist.legend(loc="upper right", fontsize=6)
    panel_letter(ax_hist, "c")

    ax_tab.axis("off")
    rows = [
        ("case / n seeds", f"index {case['index']} / {n_seeds}"),
        (r"$V_{s1}, H, V_{s2}$", f"{case['vs1']:.0f} m/s, {case['H']:.0f} m, {case['vs2']:.0f} m/s"),
        (r"$\xi$, dz", f"{case['xi']:.3f}, {cfg.dz:g} m"),
        ("cov (unused by Toro)", f"{case['cov']:.3f}"),
        (r"$\sigma_{\ln V}$ surf / at depth", f"{cfg.sigma_ln_vs_surface:.2f} / {cfg.sigma_ln_vs:.2f}"),
        (r"$\sigma_{\ln V}$ taper depth", f"{cfg.sigma_ln_vs_depth_m:.0f} m"),
        (r"inflation, clip $|Z|$", f"{cfg.toro_sigma_inflate:.2f}$\\times$, $\\leq${cfg.clip_std:g}"),
        (r"$\rho_0,\ \delta$", f"{cfg.toro_rho_0:.2f}, {cfg.toro_delta:.2f} m"),
        (r"$\rho_{200},\ b,\ h_0$", f"{cfg.toro_rho_200:.2f}, {cfg.toro_b:.3f}, {cfg.toro_h0:.0f} m"),
        ("Hallal flags", "frozen H, fixed layers/bedrock"),
    ]
    tab = ax_tab.table(
        cellText=[[k, v] for k, v in rows],
        colLabels=["Parameter", "Value"],
        cellLoc="left",
        colLoc="left",
        colWidths=[0.46, 0.54],
        bbox=(0.0, 0.0, 1.0, 0.88),
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(6.0)
    for (r, c), cell in tab.get_celld().items():
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_text_props(fontweight="bold")
    ax_tab.set_title("Config passed to `generate_vs_randomized_profile`", fontsize=7)
    panel_letter(ax_tab, "d", x=-0.02, y=1.0)

    fig.suptitle("Toro 2022 Sec. 4 frozen-H AR(1): distribution parameters", fontsize=8)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.08)
    savefig(fig, dest)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack-dir", type=Path, default=config.RESULTS_DIR / "presentation")
    p.add_argument("--domain", default="iid", choices=("iid", "dipping", "three_layer"))
    p.add_argument("--case-index", type=int, default=None)
    p.add_argument("--n-seeds", type=int, default=200)
    p.add_argument("--bedrock-view-m", type=float, default=BEDROCK_VIEW_M)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args()

    src = pack_path(args.pack_dir, args.domain)
    pack = load_pack(src)
    case = pick_case(pack, args.case_index)
    freq = np.asarray(pack["freq"], dtype=float)

    cfg, depth_mid, vs_rows, tf_rows, soil_nz = draw_ensemble(
        freq, case, n_seeds=args.n_seeds, bedrock_view_m=args.bedrock_view_m
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_vs_and_tf(
        freq, case, depth_mid, vs_rows, tf_rows, soil_nz, args.out_dir / "toro_vs_tf_demo.png"
    )
    plot_params(
        case, cfg, depth_mid, vs_rows, soil_nz, args.n_seeds, args.out_dir / "toro_params_demo.png"
    )
    print(f"Wrote {args.out_dir / 'toro_vs_tf_demo.png'}")
    print(f"Wrote {args.out_dir / 'toro_params_demo.png'}")


if __name__ == "__main__":
    main()
