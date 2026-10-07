#!/usr/bin/env python3
"""Val+test |TF| atlas: 4×4 pages sorted by window-extracted f0.

Scores GINO, 1-D Haskell nom, Pretell, Toro, and Passeri on nested IID
and dipping val, concatenates with the frozen test packs, and writes
log–log overlays. Dmult and the fixed-H Toro/Passeri arms are not drawn.
Dipping plots use the dip-depth geomeans.

    uv run python experiments/DeepONet-Residual/response_variability/plots/tf_atlas.py
    uv run python .../tf_atlas.py --skip-predict
    uv run python .../tf_atlas.py --ood-dipping-box
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_EXP = Path(__file__).resolve().parents[2]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

import config  # noqa: E402

from response_variability.covariates import attach_extracted_f0  # noqa: E402
from response_variability.names import (  # noqa: E402
    DMULT,
    DMULT_P84,
    GINO,
    HASKELL_NOMINAL,
    METHOD_COLORS,
    OPENSEES,
    PASSERI,
    PASSERI_DIP,
    PASSERI_FIXED,
    PRETELL,
    PRETELL_P84,
    TF_KEYS,
    TORO,
    TORO_DIP,
    TORO_FIXED,
)
from response_variability.metrics import band_pearson  # noqa: E402
from response_variability.plots.plot_iid import (  # noqa: E402
    _curve_at_sample,
    select_f0_quantile_indices,
)
from response_variability.plots.plot_presentation import (  # noqa: E402
    DOMAIN_SPECS,
    N_PRETELL_DEFAULT,
    attach_vs_and_pretell,
    load_domain_arrays,
    load_pack,
    pack_path,
    save_pack,
    score_gino,
)
from response_variability.style import apply_nature_style, panel_letter, savefig  # noqa: E402

ATLAS_DOMAINS = ("iid", "dipping")
ATLAS_METHODS = (
    OPENSEES,
    GINO,
    HASKELL_NOMINAL,
    PRETELL,
    TORO,
    TORO_DIP,
    PASSERI,
    PASSERI_DIP,
)
_DMULT_PACK_KEYS = (TF_KEYS[DMULT], TF_KEYS[DMULT_P84], "sigma_ln_dmult")
# High-contrast hues and long dash patterns so each curve stays readable
# on a small log–log panel. Dots are avoided: they vanish on a wiggly |TF|.
ATLAS_CURVE_STYLE = {
    OPENSEES: {"color": "#000000", "ls": "-", "lw": 2.1, "alpha": 1.0, "zorder": 5},
    HASKELL_NOMINAL: {
        "color": "#117733",
        "ls": (0, (7, 2, 1.6, 2)),
        "lw": 1.8,
        "alpha": 1.0,
        "zorder": 3,
    },
    GINO: {"color": "#0077BB", "ls": "-", "lw": 2.3, "alpha": 1.0, "zorder": 6},
    TORO: {
        "color": "#CC3311",
        "ls": (0, (7, 2.4)),
        "lw": 1.9,
        "alpha": 1.0,
        "zorder": 4,
    },
    TORO_DIP: {
        "color": "#EE6677",
        "ls": (0, (2.2, 1.4)),
        "lw": 2.0,
        "alpha": 1.0,
        "zorder": 4,
    },
    PASSERI: {
        "color": "#AA4499",
        "ls": (0, (9, 2.2, 2.2, 2.2)),
        "lw": 2.0,
        "alpha": 1.0,
        "zorder": 4,
    },
    PASSERI_DIP: {
        "color": "#882255",
        "ls": (0, (1.2, 1.3)),
        "lw": 2.0,
        "alpha": 1.0,
        "zorder": 4,
    },
    PRETELL: {
        "color": "#7A7A7A",
        "ls": (0, (0.8, 1.6)),
        "lw": 1.5,
        "alpha": 0.95,
        "zorder": 2,
    },
}
ATLAS_LABELS = {
    TORO: "Toro geomean",
    TORO_DIP: "Toro dip",
    PASSERI: "Passeri geomean",
    PASSERI_DIP: "Passeri dip",
}
PAGE_SIZE = 16
NROWS = 4
NCOLS = 4
ATLAS_YLIM = (1e-1, 5e1)
# PowerPoint 4:3 slide proportions (10x7.5 in), scaled up for legible 4x4 panels.
ATLAS_FIGSIZE = (13.333, 10.0)
_LETTERS = "abcdefghijklmnop"
OUT_DIR = config.RESULTS_DIR / "response_variability" / "tf_atlas"
PACK_DIR = config.RESULTS_DIR / "presentation"
DROP_ON_CONCAT = frozenset(
    {
        "vs_2d",
        "vs_column",
        "freq",
        "recorder_x",
        "domain",
        "n_hallal_seeds",
    }
)


def page_index_slices(n: int, page_size: int = PAGE_SIZE) -> list[np.ndarray]:
    """Partition ``0..n-1`` into contiguous pages of ``page_size`` (last may be short)."""
    n = int(n)
    if n <= 0:
        return []
    pages = []
    start = 0
    while start < n:
        stop = min(start + int(page_size), n)
        pages.append(np.arange(start, stop, dtype=int))
        start = stop
    return pages


def sort_indices_by_extracted_f0(pack: dict[str, np.ndarray]) -> np.ndarray:
    """Sample order by window-extracted ``f0`` (NaNs last)."""
    f0 = np.asarray(pack["f0"], dtype=np.float64)
    order = np.argsort(f0, kind="stable")
    nan = ~np.isfinite(f0)
    return np.concatenate([order[~nan[order]], order[nan[order]]])


def atlas_methods_in(pack: dict[str, np.ndarray]) -> list[str]:
    return [m for m in ATLAS_METHODS if TF_KEYS[m] in pack]


def _fmt_title_num(value, spec: str) -> str | None:
    x = float(value)
    if not np.isfinite(x):
        return None
    return format(x, spec)


def atlas_panel_title(pack: dict[str, np.ndarray], i: int) -> str:
    """Vs1, H, Vs2, rH, aHV, CoV, RF seed, dip (dipping), extracted f0 / split."""
    vs1 = _fmt_title_num(pack["vs1"][i], ".0f") if "vs1" in pack else None
    h = _fmt_title_num(pack["H"][i], ".0f") if "H" in pack else None
    vs2 = _fmt_title_num(pack["vs2"][i], ".0f") if "vs2" in pack else None
    line1_bits = []
    if vs1 is not None:
        line1_bits.append(rf"$V_{{s1}}$={vs1} m s$^{{-1}}$")
    if h is not None:
        line1_bits.append(rf"$H$={h} m")
    if vs2 is not None:
        line1_bits.append(rf"$V_{{s2}}$={vs2}")
    bits: list[str] = []
    if "rH" in pack:
        rh = _fmt_title_num(pack["rH"][i], ".0f")
        if rh is not None:
            bits.append(rf"$r_H$={rh} m")
    if "aHV" in pack:
        ahv = _fmt_title_num(pack["aHV"][i], ".0f")
        if ahv is not None:
            bits.append(rf"$a_{{HV}}$={ahv}")
    if "cov" in pack:
        cov = _fmt_title_num(pack["cov"][i], ".2f")
        if cov is not None:
            bits.append(rf"CoV={cov}")
    if "dip_angle_deg" in pack:
        dip = float(pack["dip_angle_deg"][i])
        if np.isfinite(dip):
            bits.append(rf"dip={dip:+.1f}$^\circ$")
    f0 = float(pack["f0"][i]) if "f0" in pack else float("nan")
    f0_s = rf"$f_0$={f0:.2f} Hz" if np.isfinite(f0) else r"$f_0$=—"
    ident: list[str] = []
    if "rf_seed" in pack:
        seed = int(pack["rf_seed"][i])
        if seed >= 0:
            ident.append(f"seed={seed}")
    ident.append(f0_s)
    if "split" in pack:
        ident.append(f"[{str(pack['split'][i])}]")
    lines = [", ".join(line1_bits)] if line1_bits else []
    if bits:
        lines.append(", ".join(bits))
    lines.append(", ".join(ident))
    return "\n".join(lines)


def attach_atlas_geometry(
    pack: dict[str, np.ndarray], domain: str
) -> dict[str, np.ndarray]:
    """Restore rH / aHV (and dipping angle) from cache meta onto a held-out pack."""
    from mix_ladder import mix_test_parts

    out = dict(pack)
    if "local_idx" in out:
        cache_dir, _idx = mix_test_parts()[DOMAIN_SPECS[domain]["mix_key"]]
        meta_path = Path(cache_dir) / "meta.npz"
        if meta_path.is_file():
            meta = dict(np.load(meta_path, allow_pickle=True))
            loc = np.asarray(out["local_idx"], dtype=int)
            for key in ("rH", "aHV"):
                if key in meta:
                    col = np.asarray(meta[key])
                    vals = np.full(loc.size, np.nan, dtype=float)
                    ok = (loc >= 0) & (loc < col.shape[0])
                    vals[ok] = np.asarray(col[loc[ok]], dtype=float)
                    out[key] = vals
            if "rf_seed" in meta:
                col = np.asarray(meta["rf_seed"])
                vals = np.full(loc.size, -1, dtype=np.int64)
                ok = (loc >= 0) & (loc < col.shape[0])
                vals[ok] = np.asarray(col[loc[ok]], dtype=np.int64)
                out["rf_seed"] = vals
    if domain == "dipping":
        out = attach_dipping_manifest(out)
    return out


def concat_heldout(
    val: dict[str, np.ndarray], test: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Stack val then test on per-sample axes; tag ``split``."""
    n_val = int(np.asarray(val["tf_opensees"]).shape[0])
    n_test = int(np.asarray(test["tf_opensees"]).shape[0])
    out: dict[str, np.ndarray] = {}
    shared = set(val) & set(test)
    for key in shared:
        if key in DROP_ON_CONCAT:
            out[key] = np.asarray(val[key])
            continue
        a = np.asarray(val[key])
        b = np.asarray(test[key])
        if a.shape[:1] == (n_val,) and b.shape[:1] == (n_test,):
            out[key] = np.concatenate([a, b], axis=0)
        else:
            out[key] = a
    for key in ("freq", "recorder_x"):
        if key in val:
            out[key] = np.asarray(val[key])
    out["split"] = np.array(["val"] * n_val + ["test"] * n_test)
    out["domain"] = np.asarray(val.get("domain", test.get("domain", "iid")))
    return out


def val_pack_path(out_dir: Path, domain: str) -> Path:
    return Path(out_dir) / f"{DOMAIN_SPECS[domain]['pack_name']}_val_pack.npz"


def heldout_pack_path(out_dir: Path, domain: str) -> Path:
    return Path(out_dir) / f"{DOMAIN_SPECS[domain]['pack_name']}_heldout_pack.npz"


def _cache_and_idx(domain: str, split: str) -> tuple[Path, np.ndarray]:
    from mix_ladder import mix_test_parts, mix_val_lookup

    mix_key = DOMAIN_SPECS[domain]["mix_key"]
    if split == "test":
        cache, idx = mix_test_parts()[mix_key]
    elif split == "val":
        cache, idx = mix_val_lookup()[mix_key]
    else:
        raise ValueError(f"split must be val or test, got {split!r}")
    return Path(cache), np.asarray(idx, dtype=int)


def attach_dipping_manifest(pack: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Fill ``dip_angle_deg`` from the OOD dipping manifest (aligned on local_idx)."""
    if "local_idx" not in pack:
        return pack
    from response_variability.sobol_cover import dipping_dip_angle_table

    table = dipping_dip_angle_table()
    if table.size == 0:
        return pack
    loc = np.asarray(pack["local_idx"], dtype=int)
    out = dict(pack)
    ok = (loc >= 0) & (loc < table.size)
    ang = np.full(loc.size, np.nan, dtype=float)
    ang[ok] = table[loc[ok]]
    out["dip_angle_deg"] = ang
    return out


def _merge_classical(pack: dict[str, np.ndarray], domain: str) -> dict[str, np.ndarray]:
    from response_variability.evals.eval_classical import merge_classical_into_pack

    return merge_classical_into_pack(pack, domain)


def load_test_pack(domain: str, *, pack_dir: Path = PACK_DIR) -> dict[str, np.ndarray]:
    path = pack_path(pack_dir, domain)
    if not path.is_file():
        raise FileNotFoundError(f"missing presentation test pack {path}")
    pack = load_pack(path)
    pack = _merge_classical(pack, domain)
    if domain == "dipping":
        pack = attach_dipping_manifest(pack)
    pack["split"] = np.full(int(pack["tf_opensees"].shape[0]), "test")
    pack["domain"] = np.array(domain)
    if "f0_calc" not in pack:
        pack = attach_extracted_f0(pack)
    return pack


def build_val_pack(
    domain: str,
    *,
    ckpt_path: Path,
    out_dir: Path,
    batch_size: int,
    n_pretell: int,
    n_hallal_seeds: int,
    skip_predict: bool,
) -> dict[str, np.ndarray]:
    dest = val_pack_path(out_dir, domain)
    if skip_predict:
        if not dest.is_file():
            raise FileNotFoundError(f"--skip-predict needs {dest}")
        pack = load_pack(dest)
        if domain == "dipping" and "dip_angle_deg" not in pack:
            pack = attach_dipping_manifest(pack)
        if "f0_calc" not in pack:
            pack = attach_extracted_f0(pack)
        return pack

    cache_dir, idx = _cache_and_idx(domain, "val")
    pack = load_domain_arrays(cache_dir, idx)
    pack["domain"] = np.array(domain)
    pack["split"] = np.full(len(idx), "val")
    pack["tf_gino"] = score_gino(cache_dir, idx, ckpt_path, batch_size)
    pack = attach_vs_and_pretell(pack, domain=domain, n_pretell=n_pretell)
    from response_variability.evals.eval_classical import add_classical_1d_arms

    pack = add_classical_1d_arms(
        pack, n_hallal_seeds=n_hallal_seeds, skip_if_present=False
    )
    if domain == "dipping":
        pack = attach_dipping_manifest(pack)
    pack = attach_extracted_f0(pack)
    save_pack(pack, dest)
    print(f"Wrote {dest}", flush=True)
    return pack


def build_heldout_pack(
    domain: str,
    *,
    ckpt_path: Path,
    out_dir: Path,
    batch_size: int,
    n_pretell: int,
    n_hallal_seeds: int,
    skip_predict: bool,
) -> dict[str, np.ndarray]:
    val = build_val_pack(
        domain,
        ckpt_path=ckpt_path,
        out_dir=out_dir,
        batch_size=batch_size,
        n_pretell=n_pretell,
        n_hallal_seeds=n_hallal_seeds,
        skip_predict=skip_predict,
    )
    test = load_test_pack(domain)
    pack = concat_heldout(val, test)
    pack = attach_atlas_geometry(pack, domain)
    if "f0_calc" not in pack:
        pack = attach_extracted_f0(pack)
    save_pack(pack, heldout_pack_path(out_dir, domain))
    return pack


def without_dmult(pack: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Atlas checks never score or plot Dmult."""
    return {key: value for key, value in pack.items() if key not in _DMULT_PACK_KEYS}


def score_heldout_csv(pack: dict[str, np.ndarray], path: Path) -> Path:
    from response_variability.evals.eval_iid import summarize_methods

    summary, peaks = summarize_methods(without_dmult(pack))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
    peaks.to_csv(path.with_name(path.stem + "_peaks.csv"), index=False)
    return path


def _atlas_style(method: str) -> dict:
    if method in ATLAS_CURVE_STYLE:
        return ATLAS_CURVE_STYLE[method]
    return {
        "color": METHOD_COLORS[method],
        "ls": "-",
        "lw": 1.2,
        "alpha": 1.0,
        "zorder": 3,
    }


def _plot_atlas_curve(ax, freq, af, method: str) -> None:
    y = np.maximum(np.asarray(af, dtype=np.float64), 1e-6)
    ax.plot(freq, y, **_atlas_style(method))


def _atlas_legend_handles(methods: list[str]):
    from matplotlib.lines import Line2D

    return [
        Line2D(
            [0],
            [0],
            color=style["color"],
            ls=style["ls"],
            lw=style["lw"],
            alpha=style["alpha"],
            label=ATLAS_LABELS.get(method, method),
        )
        for method in methods
        for style in [_atlas_style(method)]
    ]


def plot_tf_grid_4x4(
    pack: dict[str, np.ndarray],
    idx: np.ndarray,
    path: Path,
) -> Path:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    apply_nature_style()
    idx = np.asarray(idx, dtype=int)
    freq = pack["freq"]
    methods = atlas_methods_in(pack)
    tfs = {m: pack[TF_KEYS[m]] for m in methods}
    show_gino_rho = GINO in tfs and OPENSEES in tfs
    with mpl.rc_context(
        {
            "axes.titlesize": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
        }
    ):
        fig, axes = plt.subplots(
            NROWS,
            NCOLS,
            figsize=ATLAS_FIGSIZE,
            sharex=True,
            sharey=True,
        )
        axes_flat = np.asarray(axes).ravel()
        for k, ax in enumerate(axes_flat):
            if k >= len(idx):
                ax.set_visible(False)
                continue
            i = int(idx[k])
            order = [
                m for m in methods if m not in (OPENSEES, GINO, PRETELL, PRETELL_P84)
            ]
            order.extend(m for m in (PRETELL, GINO, OPENSEES) if m in tfs)
            for method in order:
                _plot_atlas_curve(ax, freq, _curve_at_sample(tfs[method], i), method)
            f0 = float(pack["f0"][i]) if "f0" in pack else float("nan")
            if np.isfinite(f0):
                ax.axvline(f0, color="0.35", ls=":", lw=0.7, zorder=1)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(0.1, 10.0)
            ax.set_ylim(*ATLAS_YLIM)
            ax.set_title(
                atlas_panel_title(pack, i), fontsize=7.5, pad=4, linespacing=1.1
            )
            panel_letter(ax, _LETTERS[k], x=0.02, y=0.93)
            if show_gino_rho:
                rho = band_pearson(
                    _curve_at_sample(tfs[GINO], i),
                    _curve_at_sample(tfs[OPENSEES], i),
                    freq,
                    lo=0.1,
                    hi=10.0,
                )
                if np.isfinite(rho):
                    ax.text(
                        0.04,
                        0.05,
                        rf"GINO–2D $\rho$={rho:.2f}",
                        transform=ax.transAxes,
                        fontsize=7,
                        ha="left",
                        va="bottom",
                        color=_atlas_style(GINO)["color"],
                        bbox={
                            "boxstyle": "round,pad=0.18",
                            "fc": "white",
                            "ec": "none",
                            "alpha": 0.75,
                        },
                    )
            if k // NCOLS == NROWS - 1 or k == len(idx) - 1:
                ax.set_xlabel(r"$f$ (Hz)")
            if k % NCOLS == 0:
                ax.set_ylabel(r"$|\mathrm{TF}|$")
        handles = _atlas_legend_handles(methods)
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=4,
            bbox_to_anchor=(0.5, 0.0),
            bbox_transform=fig.transFigure,
            frameon=False,
            handlelength=3.4,
            columnspacing=1.1,
            handletextpad=0.4,
            labelspacing=0.25,
            borderaxespad=0.2,
        )
        fig.tight_layout(h_pad=1.1, w_pad=0.6, rect=(0.0, 0.07, 1.0, 0.99))
    return savefig(fig, path)


def plot_atlas_pages(
    pack: dict[str, np.ndarray],
    out_dir: Path,
    domain: str,
    *,
    page_dir_name: str | None = None,
) -> list[Path]:
    pack = attach_atlas_geometry(without_dmult(pack), domain)
    out_dir = Path(out_dir) / (page_dir_name or domain)
    out_dir.mkdir(parents=True, exist_ok=True)
    order = sort_indices_by_extracted_f0(pack)
    pages = page_index_slices(len(order), PAGE_SIZE)
    paths: list[Path] = []
    for p_i, sl in enumerate(pages, start=1):
        idx = order[sl]
        paths.append(
            plot_tf_grid_4x4(
                pack,
                idx,
                out_dir / f"page_{p_i:02d}.png",
            )
        )
    q_idx = select_f0_quantile_indices(pack, n=PAGE_SIZE)
    paths.append(
        plot_tf_grid_4x4(
            pack,
            q_idx,
            out_dir / "tf_panels_f0_quantiles.png",
        )
    )
    return paths


def run(
    *,
    domains: tuple[str, ...] = ATLAS_DOMAINS,
    ckpt_path: Path = config.DEFAULT_CHECKPOINT,
    out_dir: Path = OUT_DIR,
    pack_dir: Path = PACK_DIR,
    batch_size: int = config.BATCH_SIZE,
    n_pretell: int = N_PRETELL_DEFAULT,
    n_hallal_seeds: int = 40,
    skip_predict: bool = False,
    skip_plot: bool = False,
) -> dict[str, list[Path]]:
    del pack_dir
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, list[Path]] = {}
    counts: dict[str, dict[str, int]] = {}
    for domain in domains:
        pack = build_heldout_pack(
            domain,
            ckpt_path=ckpt_path,
            out_dir=out_dir,
            batch_size=batch_size,
            n_pretell=n_pretell,
            n_hallal_seeds=n_hallal_seeds,
            skip_predict=skip_predict,
        )
        pack = overlay_saved_pretell(pack, domain)
        save_pack(pack, heldout_pack_path(out_dir, domain))
        csv_path = out_dir / f"{domain}_heldout_summary.csv"
        score_heldout_csv(pack, csv_path)
        split = np.asarray(pack["split"])
        counts[domain] = {
            "n_val": int(np.sum(split == "val")),
            "n_test": int(np.sum(split == "test")),
            "n_heldout": int(pack["tf_opensees"].shape[0]),
            "n_methods": len(atlas_methods_in(pack)),
        }
        paths = [] if skip_plot else plot_atlas_pages(pack, out_dir, domain)
        written[domain] = paths
        print(
            f"[{domain}] val={counts[domain]['n_val']} test={counts[domain]['n_test']} "
            f"pages={len(paths)}",
            flush=True,
        )
    counts_path = out_dir / "counts.json"
    if counts_path.is_file():
        try:
            prev = json.loads(counts_path.read_text())
        except json.JSONDecodeError:
            prev = {}
        if isinstance(prev, dict):
            counts = {**prev, **counts}
    counts_path.write_text(json.dumps(counts, indent=2) + "\n")
    iid_csv = out_dir / "iid_heldout_summary.csv"
    dip_csv = out_dir / "dipping_heldout_summary.csv"
    if iid_csv.is_file() and dip_csv.is_file():
        from response_variability.plots.plot_eval_bias import (
            plot_pearson_boxes_from_atlas,
        )

        boxes = plot_pearson_boxes_from_atlas(
            config.RESULTS_DIR
            / "response_variability"
            / "eval_bias"
            / "method_ranking_pearson_heldout.png",
            atlas_dir=out_dir,
        )
        print(f"Wrote {boxes}", flush=True)
    return written


def plot_ood_dipping_pearson(out_dir: Path) -> Path:
    """IID val+test beside dipping val+test. Dmult and fixed-H arms are omitted."""
    import pandas as pd

    from response_variability.plots.plot_eval_bias import plot_pearson_boxes_heldout

    iid = pd.read_csv(out_dir / "iid_heldout_summary.csv")
    dip = pd.read_csv(out_dir / "ood_dipping_heldout_summary.csv")
    drop = {DMULT, DMULT_P84, TORO_FIXED, PASSERI_FIXED}
    iid = iid.loc[~iid["method"].isin(drop)]
    dip = dip.loc[~dip["method"].isin(drop)]
    return plot_pearson_boxes_heldout(
        {"iid": iid, "ood_dipping": dip},
        out_dir / "ood_dipping_pearson_heldout.png",
        panels=(("iid", "IID val+test"), ("ood_dipping", "dipping val+test")),
        label_rotation=32,
    )


def _pearson_by_split(pack: dict[str, np.ndarray], summary_path: Path) -> Path:
    import pandas as pd

    summary = pd.read_csv(summary_path)
    split = np.asarray(pack["split"]).astype(str)
    summary["split"] = split[summary["sample"].to_numpy()]
    rows = []
    for (part, method), sub in summary.groupby(["split", "method"], sort=False):
        scores = sub["pearson"].to_numpy(dtype=float)
        rows.append(
            {
                "split": part,
                "method": method,
                "n": int(scores.size),
                "median": float(np.nanmedian(scores)),
                "p25": float(np.nanpercentile(scores, 25)),
                "p75": float(np.nanpercentile(scores, 75)),
            }
        )
    frame = pd.DataFrame(rows)
    dest = Path(summary_path).with_name("ood_dipping_pearson_by_split.csv")
    frame.to_csv(dest, index=False)
    print(frame.to_string(index=False), flush=True)
    return dest


def overlay_saved_pretell(
    pack: dict[str, np.ndarray], domain: str
) -> dict[str, np.ndarray]:
    """Use the Savio Pretell ensembles on Box instead of the cached spectra."""
    from response_variability.hallal_box import BOX_DIR, apply_pretell_ensemble

    if domain == "iid":
        path = BOX_DIR / "pretell_ensembles.h5"
    elif domain == "dipping":
        path = config.ood_dipping_root() / "pretell_comparison" / "pretell_ensembles.h5"
    else:
        return pack
    if not path.is_file():
        raise FileNotFoundError(f"missing Pretell ensemble {path}")
    return apply_pretell_ensemble(pack, path)


def run_ood_dipping_box(
    *, out_dir: Path = OUT_DIR, skip_plot: bool = False
) -> list[Path]:
    """Val+test atlas using Box Toro, Passeri, and Pretell geomeans."""
    from response_variability.ood_dipping_box import apply_ood_dipping_toro_passeri

    out_dir = Path(out_dir)
    src = heldout_pack_path(out_dir, "dipping")
    if not src.is_file():
        raise FileNotFoundError(f"missing dipping val+test pack {src}")
    pack = apply_ood_dipping_toro_passeri(load_pack(src))
    pack = overlay_saved_pretell(pack, "dipping")
    dest = out_dir / "ood_dipping_heldout_pack.npz"
    save_pack(pack, dest)
    print(f"Wrote {dest}", flush=True)
    csv_path = out_dir / "ood_dipping_heldout_summary.csv"
    score_heldout_csv(pack, csv_path)
    _pearson_by_split(pack, csv_path)
    split = np.asarray(pack["split"]).astype(str)
    print(
        f"[ood_dipping] val={int(np.sum(split == 'val'))} "
        f"test={int(np.sum(split == 'test'))} methods={atlas_methods_in(pack)}",
        flush=True,
    )
    paths = (
        []
        if skip_plot
        else plot_atlas_pages(pack, out_dir, "dipping", page_dir_name="ood_dipping")
    )
    if (out_dir / "iid_heldout_summary.csv").is_file():
        boxes = plot_ood_dipping_pearson(out_dir)
        print(f"Wrote {boxes}", flush=True)
        paths.append(boxes)
    counts_path = out_dir / "counts.json"
    counts: dict = {}
    if counts_path.is_file():
        try:
            prev = json.loads(counts_path.read_text())
        except json.JSONDecodeError:
            prev = {}
        if isinstance(prev, dict):
            counts = prev
    counts["ood_dipping"] = {
        "n_val": int(np.sum(split == "val")),
        "n_test": int(np.sum(split == "test")),
        "n_heldout": int(pack["tf_opensees"].shape[0]),
        "n_methods": len(atlas_methods_in(pack)),
    }
    counts_path.write_text(json.dumps(counts, indent=2) + "\n")
    return paths


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=config.DEFAULT_CHECKPOINT)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--pack-dir", type=Path, default=PACK_DIR)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--n-pretell", type=int, default=N_PRETELL_DEFAULT)
    p.add_argument("--n-hallal-seeds", type=int, default=40)
    p.add_argument(
        "--domains", nargs="+", default=list(ATLAS_DOMAINS), choices=ATLAS_DOMAINS
    )
    p.add_argument("--skip-predict", action="store_true")
    p.add_argument("--skip-plot", action="store_true")
    p.add_argument(
        "--ood-dipping-box",
        action="store_true",
        help="Atlas and val+test scores from Box ood_dipping Toro, Passeri, and Pretell H5 (no Dmult).",
    )
    args = p.parse_args()
    if args.ood_dipping_box:
        run_ood_dipping_box(out_dir=args.out_dir, skip_plot=args.skip_plot)
        return
    run(
        domains=tuple(args.domains),
        ckpt_path=args.checkpoint,
        out_dir=args.out_dir,
        pack_dir=args.pack_dir,
        batch_size=args.batch_size,
        n_pretell=args.n_pretell,
        n_hallal_seeds=args.n_hallal_seeds,
        skip_predict=args.skip_predict,
        skip_plot=args.skip_plot,
    )


if __name__ == "__main__":
    main()
