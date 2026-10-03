#!/usr/bin/env python3
"""Vs realizations and |TF| for the dipping uniform-depth Toro and Passeri models.

One steep val case. Each curve is one soil randomization at one depth from the
uniform dip grid. Vs2 is the case value at every depth. Passeri adds the
travel-time profile, which is the quantity it actually draws.

    uv run python experiments/DeepONet-Residual/response_variability/plots/plot_dip_uniform_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_EXP = Path(__file__).resolve().parents[2]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

import config  # noqa: E402

from haskell_baseline import haskell_af_within  # noqa: E402
from response_variability.dip_depth import (  # noqa: E402
    DIP_SPAN_M,
    depth_range,
    sigma_y,
    uniform_dip_depths,
)
from response_variability.metrics import central_recorder  # noqa: E402
from response_variability.names import HASKELL_NOMINAL, METHOD_COLORS, PASSERI, TORO  # noqa: E402
from response_variability.seiskit_arms import ensure_seiskit, hallal_config  # noqa: E402
from response_variability.style import (  # noqa: E402
    apply_nature_style,
    figsize,
    panel_letter,
    savefig,
)
from response_variability.plots.tf_atlas import OUT_DIR, atlas_panel_title  # noqa: E402

PACK = OUT_DIR / "dipping_uniform_pack.npz"
FIG_DIR = OUT_DIR / "dipping"
DZ = 0.5
N_SHOW = 24
BEDROCK_VIEW_M = 8.0


def pick_steep_val(pack: dict[str, np.ndarray]) -> int:
    split = np.asarray(pack["split"]).astype(str)
    angle = np.abs(np.asarray(pack["dip_angle_deg"], dtype=np.float64))
    angle = np.where(split == "val", angle, -np.inf)
    return int(np.argmax(angle))


def _draw_kind(
    freq: np.ndarray,
    *,
    kind: str,
    vs1: float,
    depths: np.ndarray,
    cov: float,
    vs2: float,
    xi: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray]:
    """One seed per uniform depth. Returns z, Vs, cumulative tt, and |TF| rows."""
    ensure_seiskit()
    from seiskit.profile_randomization import (
        generate_tts_randomized_profile,
        generate_vs_randomized_profile,
    )

    gen = (
        generate_vs_randomized_profile
        if kind == "toro"
        else generate_tts_randomized_profile
    )
    z_rows: list[np.ndarray] = []
    vs_rows: list[np.ndarray] = []
    tt_rows: list[np.ndarray] = []
    tf_rows = np.empty((depths.size, freq.shape[0]), dtype=np.float64)
    for k, depth in enumerate(depths):
        cfg = hallal_config(vs1=vs1, H=float(depth), cov=cov, vs2=vs2, dz=DZ)
        rng = np.random.default_rng(k + 1)
        vs = np.asarray(gen(cfg, rng), dtype=float)
        soil_nz = max(1, int(round(float(depth) / DZ)))
        soil_nz = min(soil_nz, vs.size)
        z = (np.arange(soil_nz) + 0.5) * DZ
        soil = vs[:soil_nz]
        z_rows.append(z)
        vs_rows.append(soil)
        tt_rows.append(np.cumsum(DZ / np.maximum(soil, 1e-6)))
        zeta = np.full(vs.size, xi)
        tf_rows[k] = haskell_af_within(
            freq, vs, zeta, dz=DZ, vs_rock=vs2, soil_nz=soil_nz
        )
    return z_rows, vs_rows, tt_rows, tf_rows


def _case_fields(pack: dict[str, np.ndarray], i: int) -> dict[str, float]:
    xi = config.DEFAULT_XI_TREND
    if (
        "xi_mean" in pack
        and np.isfinite(pack["xi_mean"][i])
        and pack["split"][i] == "test"
    ):
        xi = float(pack["xi_mean"][i])
    return {
        "vs1": float(pack["vs1"][i]),
        "H": float(pack["H"][i]),
        "cov": float(pack["cov"][i]),
        "vs2": float(pack["vs2"][i]),
        "theta": float(pack["dip_angle_deg"][i]),
        "xi": float(xi),
    }


def _plot_profiles(
    ax, z_rows, vs_rows, *, color: str, H: float, vs1: float, vs2: float
) -> None:
    for z, vs in zip(z_rows, vs_rows):
        ax.step(vs, z, color=color, alpha=0.28, lw=0.7, where="mid")
        z_iface = float(z[-1] + 0.5 * DZ)
        ax.plot(
            [vs[-1], vs2, vs2],
            [z_iface, z_iface, z_iface + BEDROCK_VIEW_M],
            color=color,
            alpha=0.28,
            lw=0.7,
        )
    z_nom = np.array([0.0, H, H, H + BEDROCK_VIEW_M])
    vs_nom = np.array([vs1, vs1, vs2, vs2])
    ax.plot(
        vs_nom,
        z_nom,
        color=METHOD_COLORS[HASKELL_NOMINAL],
        ls="-.",
        lw=1.15,
        label=HASKELL_NOMINAL,
        zorder=4,
    )
    ax.set_xlabel(r"$V_s$ (m/s)")
    ax.set_ylabel("Depth (m)")
    deepest = max(float(z[-1]) for z in z_rows) + BEDROCK_VIEW_M
    ax.set_ylim(deepest + DZ, 0.0)
    ax.set_xlim(0.0, vs2 * 1.08)


def _plot_tf(
    ax, freq, tf_rows, pack, i, *, color: str, geomean_key: str, label: str
) -> None:
    ops = np.asarray(pack["tf_opensees"][i], dtype=np.float64)
    for r in range(ops.shape[0]):
        ax.loglog(
            freq,
            np.maximum(ops[r], 1e-6),
            color="0.72",
            lw=0.6,
            zorder=1,
            label="2D stations" if r == 0 else None,
        )
    for row in tf_rows:
        ax.loglog(
            freq, np.maximum(row, 1e-6), color=color, alpha=0.22, lw=0.55, zorder=2
        )
    geo = np.maximum(np.asarray(pack[geomean_key][i], dtype=np.float64), 1e-6)
    nom = np.maximum(central_recorder(pack["tf_haskell_nominal"][i]), 1e-6)
    ax.loglog(freq, geo, color=color, lw=1.6, zorder=4, label=label)
    ax.loglog(
        freq,
        nom,
        color=METHOD_COLORS[HASKELL_NOMINAL],
        ls="-.",
        lw=1.15,
        zorder=3,
        label=HASKELL_NOMINAL,
    )
    ax.set_xlim(0.1, 10.0)
    ax.set_ylim(1e-1, 5e1)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"$|\mathrm{TF}|$")
    ax.legend(loc="lower left", fontsize=6)


def plot_toro(pack, i: int, freq, fields, z_rows, vs_rows, tf_rows, dest: Path) -> None:
    import matplotlib.pyplot as plt

    apply_nature_style()
    color = METHOD_COLORS[TORO]
    fig, (ax_vs, ax_tf) = plt.subplots(1, 2, figsize=figsize("double", height_mm=78))
    _plot_profiles(
        ax_vs,
        z_rows,
        vs_rows,
        color=color,
        H=fields["H"],
        vs1=fields["vs1"],
        vs2=fields["vs2"],
    )
    half = 0.5 * depth_range(DIP_SPAN_M, fields["theta"])
    ax_vs.axhspan(
        fields["H"] - half, fields["H"] + half, color=color, alpha=0.08, lw=0, zorder=0
    )
    ax_vs.set_title(f"seiskit frozen-H Toro, NHPP off (n={len(z_rows)})")
    panel_letter(ax_vs, "a")
    _plot_tf(
        ax_tf,
        freq,
        tf_rows,
        pack,
        i,
        color=color,
        geomean_key="tf_toro",
        label=f"{TORO} geomean",
    )
    ax_tf.set_title("Transfer function")
    panel_letter(ax_tf, "b")
    sig = sigma_y(DIP_SPAN_M, fields["theta"], n=500)
    fig.suptitle(
        "Toro, uniform dip depth — "
        + atlas_panel_title(pack, i).replace("\n", ", ")
        + rf", $\sigma_y$={sig:.2f} m",
        fontsize=7.5,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    savefig(fig, dest)


def plot_passeri(
    pack, i: int, freq, fields, z_rows, vs_rows, tt_rows, tf_rows, dest: Path
) -> None:
    import matplotlib.pyplot as plt

    apply_nature_style()
    color = METHOD_COLORS[PASSERI]
    fig, (ax_vs, ax_tt, ax_tf) = plt.subplots(
        1, 3, figsize=figsize("double", height_mm=78)
    )
    _plot_profiles(
        ax_vs,
        z_rows,
        vs_rows,
        color=color,
        H=fields["H"],
        vs1=fields["vs1"],
        vs2=fields["vs2"],
    )
    ax_vs.set_title(f"Uniform depths, one Passeri $V_s$ each (n={len(z_rows)})")
    panel_letter(ax_vs, "a")

    for z, tt in zip(z_rows, tt_rows):
        ax_tt.plot(tt, z, color=color, alpha=0.35, lw=0.7)
    z_nom = np.linspace(0.0, fields["H"], 40)
    ax_tt.plot(
        z_nom / fields["vs1"],
        z_nom,
        color=METHOD_COLORS[HASKELL_NOMINAL],
        ls="-.",
        lw=1.15,
        label=HASKELL_NOMINAL,
    )
    deepest = max(float(z[-1]) for z in z_rows) + 1.0
    ax_tt.set_ylim(deepest, 0.0)
    ax_tt.set_xlabel(r"Travel time $t(z)$ (s)")
    ax_tt.set_ylabel("Depth (m)")
    ax_tt.set_title(r"Soil travel time $t(z)=\int dz/V_s$")
    ax_tt.legend(loc="lower left", fontsize=6)
    panel_letter(ax_tt, "b")

    _plot_tf(
        ax_tf,
        freq,
        tf_rows,
        pack,
        i,
        color=color,
        geomean_key="tf_passeri",
        label=f"{PASSERI} geomean",
    )
    ax_tf.set_title("Transfer function")
    panel_letter(ax_tf, "c")
    fig.suptitle(
        "Passeri, uniform dip depth — "
        + atlas_panel_title(pack, i).replace("\n", ", "),
        fontsize=7.5,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    savefig(fig, dest)


def main() -> None:
    pack = dict(np.load(PACK, allow_pickle=True))
    i = pick_steep_val(pack)
    fields = _case_fields(pack, i)
    freq = np.asarray(pack["freq"], dtype=np.float64)
    depths = uniform_dip_depths(fields["H"], fields["theta"], L_h=DIP_SPAN_M, n=N_SHOW)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    z_t, vs_t, _tt_t, tf_t = _draw_kind(
        freq,
        kind="toro",
        vs1=fields["vs1"],
        depths=depths,
        cov=fields["cov"],
        vs2=fields["vs2"],
        xi=fields["xi"],
    )
    toro_path = FIG_DIR / "toro_uniform_vs_tf.png"
    plot_toro(pack, i, freq, fields, z_t, vs_t, tf_t, toro_path)

    z_p, vs_p, tt_p, tf_p = _draw_kind(
        freq,
        kind="passeri",
        vs1=fields["vs1"],
        depths=depths,
        cov=fields["cov"],
        vs2=fields["vs2"],
        xi=fields["xi"],
    )
    pas_path = FIG_DIR / "passeri_uniform_vs_tf_tts.png"
    plot_passeri(pack, i, freq, fields, z_p, vs_p, tt_p, tf_p, pas_path)
    print(f"Wrote {toro_path}")
    print(f"Wrote {pas_path}")


if __name__ == "__main__":
    main()
