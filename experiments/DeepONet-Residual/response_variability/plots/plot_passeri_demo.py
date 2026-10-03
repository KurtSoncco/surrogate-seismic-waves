#!/usr/bin/env python3
"""Passeri tts randomization demo: same idea as plot_toro_demo, but Passeri randomizes a
travel time, not Vs directly -- so the params figure gets an extra panel for it.

Draws an ensemble of frozen-H Passeri realizations for one real IID case (same call
seiskit_arms.hallal_geomean_tf makes with ``kind="passeri"``: hallal_config ->
generate_tts_randomized_profile, one draw per seed), runs each through the
Thomson-Haskell 1-D solver, and makes two figures:

  passeri_vs_tf_demo.png
      Two panels: randomized Vs(z) profiles (median +/- 1 log-std) and the
      resulting |TF(f)| ensemble (geomean +/- 1 log-std), for the same case.

  passeri_params_demo.png
      What actually drives the randomization for that case:
        a. the travel-time Z marginal (the quantity Passeri actually draws), vs its
           clipped-normal target;
        b. the Vs marginal that falls out of inverting thickness/travel-time;
        c. that Vs spread next to a Toro draw on the *same* case, since Passeri's
           sigma_ln_tts is ~10x tighter than Toro's sigma_ln_vs;
        d. a table of every ``ProfileRandomizationConfig`` field the draw touches.

Frozen-H detail worth knowing before reading the plots: seiskit's simplified Passeri
path (``generate_tts_simplified``) finds soil "layers" by looking for Vs jumps in the
flat nominal column. Since that column is flat (one Vs1 value down to H), it always
finds exactly one soil layer -- so the randomization below is a single travel-time
draw for the whole column, not a depth-resolved field like Toro's per-dz AR(1). The
Vs(z) panel is a set of flat steps for that reason, not a plotting bug.

Usage: python -m response_variability.plots.plot_passeri_demo [--case-index N] [--n-seeds N]
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
from response_variability.names import HASKELL_NOMINAL, METHOD_COLORS, PASSERI, TORO  # noqa: E402
from response_variability.plots.plot_presentation import load_pack, pack_path  # noqa: E402
from response_variability.plots.plot_toro_demo import (  # noqa: E402
    BEDROCK_VIEW_M,
    DZ,
    LN_STD_Z,
    OUT_DIR,
    pick_case,
)
from response_variability.seiskit_arms import ensure_seiskit, hallal_config  # noqa: E402
from response_variability.style import (  # noqa: E402
    apply_nature_style,
    figsize,
    panel_letter,
    savefig,
)


def draw_ensemble(
    freq: np.ndarray,
    case: dict[str, float | int],
    *,
    n_seeds: int,
    dz: float = DZ,
    bedrock_view_m: float = BEDROCK_VIEW_M,
):
    """Return (cfg, depth_mid, vs_rows[n_seeds, n_soil+n_bedrock], tf_rows[n_seeds, n_freq], soil_nz).

    Same shape contract as ``plot_toro_demo.draw_ensemble``, but the generator is
    ``generate_tts_randomized_profile``. In frozen-H mode it fills the whole soil
    column with one Vs value per seed (see module docstring), so ``vs_rows`` is
    depth-constant within ``[:soil_nz]`` for every row -- that's expected here.
    """
    ensure_seiskit()
    from seiskit.profile_randomization import generate_tts_randomized_profile

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
        vs = np.asarray(generate_tts_randomized_profile(cfg, rng), dtype=float)
        zeta = np.full_like(vs, float(case["xi"]))
        tf_rows[k] = haskell_af_within(
            freq, vs, zeta, dz=dz, vs_rock=case["vs2"], soil_nz=soil_nz
        )
        vs_rows[k] = vs[:n_total]
    depth_mid = (np.arange(n_total) + 0.5) * dz
    return cfg, depth_mid, vs_rows, tf_rows, soil_nz


def draw_toro_soil_vs(
    case: dict[str, float | int], *, n_seeds: int, dz: float = DZ
) -> np.ndarray:
    """Surface-sample Vs from a Toro draw on the same case, for the sigma comparison panel."""
    ensure_seiskit()
    from seiskit.profile_randomization import generate_vs_randomized_profile

    cfg = hallal_config(
        vs1=case["vs1"], H=case["H"], cov=case["cov"], vs2=case["vs2"], dz=dz
    )
    out = np.empty(n_seeds, dtype=np.float64)
    for k, seed in enumerate(range(1, n_seeds + 1)):
        rng = np.random.default_rng(seed)
        vs = np.asarray(generate_vs_randomized_profile(cfg, rng), dtype=float)
        out[k] = vs[0]
    return out


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
    color = METHOD_COLORS[PASSERI]
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
    show_idx = rng_show.choice(
        vs_rows.shape[0], size=min(n_show, vs_rows.shape[0]), replace=False
    )
    for i in show_idx:
        ax_vs.step(vs_rows[i], depth_mid, color=color, alpha=0.2, lw=0.6, where="mid")
    ax_vs.fill_betweenx(
        depth_mid,
        vs_lo,
        vs_hi,
        color=color,
        alpha=0.3,
        lw=0,
        label=r"median $\pm\,1\sigma_{\ln}$",
    )
    ax_vs.step(
        vs_median,
        depth_mid,
        color=color,
        lw=1.4,
        where="mid",
        label="median (randomized)",
    )
    nominal_vs = np.where(depth_mid <= case["H"], case["vs1"], case["vs2"])
    ax_vs.step(
        nominal_vs,
        depth_mid,
        color=base_color,
        ls="-.",
        lw=1.2,
        where="mid",
        label=f"{HASKELL_NOMINAL} ($V_{{s1}}$/$V_{{s2}}$)",
    )
    ax_vs.axhline(case["H"], color="0.6", ls=":", lw=0.7)
    ax_vs.annotate(
        "soil / bedrock",
        (0.02, case["H"]),
        xycoords=("axes fraction", "data"),
        textcoords="offset points",
        xytext=(0, 3),
        fontsize=5.5,
        color="0.4",
    )
    # Same soil-only zoom as the Toro demo. The band is visibly much narrower here --
    # that's real: sigma_ln_tts is ~0.02 vs Toro's depth-tapered 0.15-0.25.
    soil_lo = float(np.min(vs_lo[:soil_nz]))
    soil_hi = float(np.max(vs_hi[:soil_nz]))
    pad = max(0.12 * (soil_hi - soil_lo), 1.0)
    ax_vs.set_xlim(soil_lo - pad, soil_hi + pad)
    ax_vs.annotate(
        f"bedrock $V_{{s2}}$={case['vs2']:.0f} m/s →",
        (0.98, case["H"] + 0.06 * (depth_mid[-1] - depth_mid[0])),
        xycoords=("axes fraction", "data"),
        ha="right",
        fontsize=5.5,
        color=base_color,
    )
    ax_vs.set_ylim(depth_mid[-1] + DZ, 0.0)
    ax_vs.set_xlabel(r"$V_s$ (m/s)")
    ax_vs.set_ylabel("Depth (m)")
    ax_vs.set_title(f"Randomized $V_s(z)$ (flat per draw), n={vs_rows.shape[0]} seeds")
    ax_vs.legend(loc="lower left", fontsize=6)
    panel_letter(ax_vs, "a")

    for i in show_idx:
        ax_tf.loglog(
            freq, np.maximum(tf_rows[i], 1e-6), color=color, alpha=0.15, lw=0.5
        )
    ax_tf.fill_between(
        freq,
        np.maximum(tf_lo, 1e-6),
        np.maximum(tf_hi, 1e-6),
        color=color,
        alpha=0.3,
        lw=0,
        label=r"median $\pm\,1\sigma_{\ln}$",
    )
    ax_tf.loglog(
        freq,
        np.maximum(tf_geomean, 1e-6),
        color=color,
        lw=1.4,
        label=f"{PASSERI} geomean",
    )
    ax_tf.loglog(
        freq,
        np.maximum(nominal_tf, 1e-6),
        color=base_color,
        ls="-.",
        lw=1.2,
        label=HASKELL_NOMINAL,
    )
    ax_tf.set_xlabel("Frequency (Hz)")
    ax_tf.set_ylabel(r"$|\mathrm{TF}|$")
    ax_tf.set_title("Transfer function ensemble")
    ax_tf.legend(loc="lower left", fontsize=6)
    panel_letter(ax_tf, "b")

    fig.suptitle(
        f"Passeri tts (frozen-H) randomization -- case {case['index']}: "
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
    vs_rows: np.ndarray,
    soil_nz: int,
    toro_soil_vs: np.ndarray,
    n_seeds: int,
    dest: Path,
) -> None:
    import matplotlib.pyplot as plt

    apply_nature_style()
    color = METHOD_COLORS[PASSERI]
    toro_color = METHOD_COLORS[TORO]

    # Passeri's actual random variable: the soil column's total travel time (one
    # value per seed in frozen-H mode -- soil_vs is depth-constant, see docstring).
    soil_vs = vs_rows[:, 0]
    tt_nominal = case["H"] / case["vs1"]
    tt_seed = case["H"] / soil_vs
    tts_z = (np.log(tt_seed) - np.log(tt_nominal)) / cfg.sigma_ln_tts

    ln_vs = np.log(soil_vs / case["vs1"])
    vs_sigma_ln = float(np.std(ln_vs, ddof=1))
    toro_sigma_ln = float(np.std(np.log(toro_soil_vs / case["vs1"]), ddof=1))

    fig, axes = plt.subplots(
        2,
        2,
        figsize=figsize("double", height_mm=140),
        gridspec_kw={"hspace": 0.65, "wspace": 0.3},
    )
    ax_tt, ax_vs, ax_bar, ax_tab = axes.ravel()

    bins_tt = np.linspace(-3.0, 3.0, 25)
    ax_tt.hist(
        tts_z,
        bins=bins_tt,
        density=True,
        color=color,
        alpha=0.55,
        label="drawn $Z$ (tts)",
    )
    zz = np.linspace(-3.0, 3.0, 400)
    ax_tt.plot(
        zz,
        np.exp(-0.5 * zz**2) / np.sqrt(2 * np.pi),
        color="0.2",
        lw=1.0,
        label=r"$N(0,1)$",
    )
    ax_tt.axvline(cfg.clip_std, color="0.3", ls="--", lw=0.8)
    ax_tt.axvline(
        -cfg.clip_std,
        color="0.3",
        ls="--",
        lw=0.8,
        label=f"clip $|Z|\\leq${cfg.clip_std:g}",
    )
    ax_tt.set_xlabel(
        r"$Z=\ln(tt/tt_{\mathrm{nom}})/\sigma_{\ln\,tts}$, whole soil layer"
    )
    ax_tt.set_ylabel("Density")
    ax_tt.set_title(f"Travel-time Z (what Passeri actually draws), n={n_seeds}")
    ax_tt.legend(loc="upper right", fontsize=6)
    panel_letter(ax_tt, "a")

    span = max(4.0 * vs_sigma_ln, 1e-3)
    xs = np.linspace(-span, span, 400)
    ax_vs.hist(
        ln_vs,
        bins=25,
        density=True,
        color=color,
        alpha=0.55,
        range=(-span, span),
        label="drawn $\\ln(V_s/V_{s1})$",
    )
    ax_vs.plot(
        xs,
        np.exp(-0.5 * (xs / vs_sigma_ln) ** 2) / (vs_sigma_ln * np.sqrt(2 * np.pi)),
        color="0.2",
        lw=1.0,
        label=f"fit $N(0,{vs_sigma_ln:.3f}^2)$",
    )
    ax_vs.set_xlabel(r"$\ln(V_s/V_{s1})$, whole soil layer")
    ax_vs.set_ylabel("Density")
    ax_vs.set_title("Vs after $V_s=H/tt$ inversion")
    ax_vs.legend(loc="upper right", fontsize=6)
    panel_letter(ax_vs, "b")

    bars = ax_bar.bar(
        [TORO, PASSERI],
        [toro_sigma_ln, vs_sigma_ln],
        color=[toro_color, color],
        width=0.6,
    )
    for rect, val in zip(bars, [toro_sigma_ln, vs_sigma_ln]):
        ax_bar.annotate(
            f"{val:.3f}",
            (rect.get_x() + rect.get_width() / 2, val),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=6.5,
        )
    ax_bar.set_ylabel(r"empirical $\sigma_{\ln V_s}$ (surface sample)")
    ax_bar.set_title("Same case, same n seeds: effective Vs spread")
    panel_letter(ax_bar, "c")

    ax_tab.axis("off")
    rows = [
        ("case / n seeds", f"index {case['index']} / {n_seeds}"),
        (
            r"$V_{s1}, H, V_{s2}$",
            f"{case['vs1']:.0f} m/s, {case['H']:.0f} m, {case['vs2']:.0f} m/s",
        ),
        (r"$\xi$, dz", f"{case['xi']:.3f}, {cfg.dz:g} m"),
        ("cov (feeds safety-clip $\\sigma$)", f"{case['cov']:.3f}"),
        (r"$\sigma_{\ln\,tts}$ (the actual draw)", f"{cfg.sigma_ln_tts:.3f}"),
        (r"tts $\rho$ boost", f"+{cfg.tts_rho_boost:.2f} (moot: 1 layer)"),
        (
            r"$V_s$ safety clip $\sigma$",
            f"max({cfg.sigma_ln_vs:.2f}, $\\sqrt{{\\ln(1+cov^2)}}$)",
        ),
        ("clip $|Z|\\leq$", f"{cfg.clip_std:g}"),
        ("tt bound (factor)", "$\\times$3 either way"),
        ("soil layers detected", "1 (flat base, no Vs jump)"),
        ("Hallal flags", "frozen H, fixed layers/bedrock"),
    ]
    tab = ax_tab.table(
        cellText=[[k, v] for k, v in rows],
        colLabels=["Parameter", "Value"],
        cellLoc="left",
        colLoc="left",
        colWidths=[0.5, 0.5],
        bbox=(0.0, 0.0, 1.0, 0.88),
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(6.0)
    for (r, c), cell in tab.get_celld().items():
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_text_props(fontweight="bold")
    ax_tab.set_title("Config passed to `generate_tts_randomized_profile`", fontsize=7)
    panel_letter(ax_tab, "d", x=-0.02, y=1.0)

    fig.suptitle("Passeri tts (frozen-H): distribution parameters", fontsize=8)
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
    toro_soil_vs = draw_toro_soil_vs(case, n_seeds=args.n_seeds)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_vs_and_tf(
        freq,
        case,
        depth_mid,
        vs_rows,
        tf_rows,
        soil_nz,
        args.out_dir / "passeri_vs_tf_demo.png",
    )
    plot_params(
        case,
        cfg,
        vs_rows,
        soil_nz,
        toro_soil_vs,
        args.n_seeds,
        args.out_dir / "passeri_params_demo.png",
    )
    print(f"Wrote {args.out_dir / 'passeri_vs_tf_demo.png'}")
    print(f"Wrote {args.out_dir / 'passeri_params_demo.png'}")


if __name__ == "__main__":
    main()
