#!/usr/bin/env python3
"""Sample Pretell, Toro, and damping-multiplier curves against OpenSees.

Horizontal is the IID held-out pack. Dipping is the dipping held-out pack,
which is the one that stores all three arms. Scores are the atlas summaries:
Pearson of |TF| and Anderson misfit, center station, 0.1–10 Hz.

    python plot_classical.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PAPER = REPO / "experiments" / "DeepONet-Residual" / "paper"
ATLAS = REPO / "experiments" / "DeepONet-Residual" / "results" / "response_variability" / "tf_atlas"
sys.path.insert(0, str(PAPER))

from style import (  # noqa: E402
    METHOD_STYLE,
    OPENSEES,
    figsize,
    panel_letter,
    plot_curve,
    use_style,
)

# Atlas summaries use the seiskit names. Paper style keys differ for Toro.
PRETELL = "Pretell median"
TORO = "Toro Vs"
DMULT = "Dmult"
STYLE = {
    PRETELL: METHOD_STYLE["Pretell median"],
    TORO: METHOD_STYLE["Toro"],
    DMULT: METHOD_STYLE["Dmult"],
}

FIG_DIR = HERE / "figures"
METHODS = (PRETELL, TORO, DMULT)
PACKS = {
    "Horizontal": ATLAS / "iid_heldout_pack.npz",
    "Dipping": ATLAS / "dipping_heldout_pack.npz",
}
SUMMARIES = {
    "Horizontal": ATLAS / "iid_heldout_summary.csv",
    "Dipping": ATLAS / "dipping_heldout_summary.csv",
}


def _save(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.pdf", dpi=300)
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300)
    plt.close(fig)


def _center(tf: np.ndarray) -> np.ndarray:
    a = np.asarray(tf, dtype=np.float64)
    return a if a.ndim == 1 else a[a.shape[0] // 2]


def _sample_index(pack: np.lib.npyio.NpzFile, k: int = 6) -> np.ndarray:
    """One record from each of k soils spread in Vs1 and thickness."""
    vs1 = np.round(np.asarray(pack["vs1"], dtype=float)).astype(int)
    h = np.round(np.asarray(pack["H"], dtype=float)).astype(int)
    key = vs1.astype(np.int64) * 1000 + h.astype(np.int64)
    order = np.argsort(key)
    first = order[np.concatenate([[True], key[order][1:] != key[order][:-1]])]
    first = first[np.argsort(key[first])]
    if first.size <= k:
        return first
    return first[np.linspace(0, first.size - 1, k).round().astype(int)]


def _gallery(name: str) -> None:
    pack = np.load(PACKS[name], allow_pickle=True)
    freq = np.asarray(pack["freq"], dtype=float)
    idx = _sample_index(pack)
    use_style()
    fig, axes = plt.subplots(2, 3, figsize=figsize("double", height_mm=118), sharex=True, sharey=True)
    series = (
        ("tf_opensees", METHOD_STYLE[OPENSEES], "OpenSees"),
        ("tf_pretell", STYLE[PRETELL], "Pretell"),
        ("tf_toro", STYLE[TORO], "Toro"),
        ("tf_dmult", STYLE[DMULT], "Dmult"),
    )
    for ax, i, letter in zip(axes.ravel(), idx, "abcdef"):
        for key, style, label in series:
            plot_curve(ax, freq, np.maximum(_center(pack[key][int(i)]), 1e-6), style, label=label)
        vs1 = float(pack["vs1"][int(i)])
        h = float(pack["H"][int(i)])
        ax.set_title(f"{name}  $V_{{s1}}$={vs1:.0f}, $H$={h:.0f} m", fontsize=7, pad=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(float(freq[0]), float(freq[-1]))
        panel_letter(ax, letter)
    for ax in axes[1]:
        ax.set_xlabel(r"$f$ (Hz)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$|$TF$|$")
    axes[0, 0].legend(frameon=False, fontsize=6, loc="upper right")
    fig.tight_layout()
    _save(fig, "fig_tf_classical_" + name.lower())


def _arm_scores(pack_path: Path, key: str, f0: np.ndarray) -> dict[str, np.ndarray]:
    """Center-station scores, same definitions as the atlas summary."""
    pack = np.load(pack_path, allow_pickle=True)
    freq = np.asarray(pack["freq"], dtype=float)
    ops = np.asarray(pack["tf_opensees"], dtype=float)
    arm = np.asarray(pack[key], dtype=float)
    n = ops.shape[0]
    pear = np.empty(n)
    gof = np.empty(n)
    rel = np.empty(n)
    for i in range(n):
        ref = np.clip(_center(ops[i]), 1e-12, None)
        cand = np.clip(_center(arm[i]), 1e-12, None)
        pear[i] = float(np.corrcoef(cand, ref)[0, 1])
        rel[i] = float(np.linalg.norm(cand - ref) / max(np.linalg.norm(ref), 1e-12))
        w = np.exp(-0.5 * ((freq - float(f0[i])) / 1.5) ** 2)
        w = w / max(w.sum(), 1e-12)
        gof[i] = float(np.sum(w * np.abs(np.log(cand) - np.log(ref))))
    return {"pearson": pear, "gof_af": gof, "rel_l2": rel}


def _domain_frame(domain: str) -> pd.DataFrame:
    df = pd.read_csv(SUMMARIES[domain])
    base = df[df["method"].isin((PRETELL, TORO))].copy()
    f0 = (
        df.loc[df["method"] == PRETELL, ["sample", "f0"]]
        .drop_duplicates("sample")
        .sort_values("sample")["f0"]
        .to_numpy()
    )
    check = _arm_scores(PACKS[domain], "tf_pretell", f0)
    published = df.loc[df["method"] == PRETELL].sort_values("sample")
    err_p = float(np.max(np.abs(check["pearson"] - published["pearson"].to_numpy())))
    err_g = float(np.max(np.abs(check["gof_af"] - published["gof_af"].to_numpy())))
    if err_p > 1e-6 or err_g > 1e-5:
        raise RuntimeError(f"{domain} score mismatch pearson {err_p} gof {err_g}")
    scored = _arm_scores(PACKS[domain], "tf_dmult", f0)
    extra = pd.DataFrame({"method": DMULT, "sample": np.arange(f0.size), **scored})
    return pd.concat([base, extra], ignore_index=True)


def _stats(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for domain, df in frames.items():
        for method, g in df.groupby("method"):
            rows.append(
                {
                    "domain": domain,
                    "method": method,
                    "n": int(len(g)),
                    "pearson_median": float(g["pearson"].median()),
                    "pearson_q25": float(g["pearson"].quantile(0.25)),
                    "pearson_q75": float(g["pearson"].quantile(0.75)),
                    "gof_median": float(g["gof_af"].median()),
                    "rel_l2_median": float(g["rel_l2"].median()),
                }
            )
    return pd.DataFrame(rows)


def _boxes(frames: dict[str, pd.DataFrame], table: pd.DataFrame) -> None:
    use_style()
    labels, pear, gof, colors = [], [], [], []
    for domain in ("Horizontal", "Dipping"):
        df = frames[domain]
        short = "Horiz." if domain == "Horizontal" else "Dip."
        for method in METHODS:
            g = df.loc[df["method"] == method]
            labels.append(f"{short}\n{method.split()[0]}")
            pear.append(g["pearson"].to_numpy())
            gof.append(g["gof_af"].to_numpy())
            colors.append(STYLE[method]["color"])
    fig, axes = plt.subplots(1, 2, figsize=figsize("double", height_mm=92))
    for ax, data, ylabel, letter in (
        (axes[0], pear, r"Pearson of $|$TF$|$", "a"),
        (axes[1], gof, "Anderson misfit", "b"),
    ):
        bp = ax.boxplot(
            data,
            widths=0.62,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "lw": 0.8},
            whiskerprops={"lw": 0.6},
            capprops={"lw": 0.6},
            boxprops={"lw": 0.6},
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
            patch.set_edgecolor("#333333")
        ax.set_xticks(range(1, len(labels) + 1), labels, fontsize=6)
        ax.set_ylabel(ylabel)
        panel_letter(ax, letter, x=-0.16)
    axes[0].set_ylim(0.55, 1.02)
    fig.tight_layout()
    _save(fig, "fig_box_classical")
    table.to_csv(FIG_DIR / "classical_medians.csv", index=False)


def main() -> None:
    for name in PACKS:
        _gallery(name)
    frames = {name: _domain_frame(name) for name in PACKS}
    table = _stats(frames)
    _boxes(frames, table)
    print(table.to_string(index=False), flush=True)
    print(f"wrote {FIG_DIR}", flush=True)


if __name__ == "__main__":
    main()
