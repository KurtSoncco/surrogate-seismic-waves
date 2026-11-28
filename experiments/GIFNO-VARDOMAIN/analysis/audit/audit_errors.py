#!/usr/bin/env python3
"""Phase 1.8 frozen P128n128 error against the coverage coordinates.

IID and dipping center stations are the run_g0 practice tables. Three-layer,
manifold, and any family those tables do not cover are scored with load_fp32
only. The checkpoint is not fine-tuned and no second dtype is loaded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from common import (
    AUDIT, BAND_NAMES, CKPT, REPO, bootstrap_independent, provenance, save_fig, use_agg,
)
from score_harness import aggregate, harm_rate, score_curves

COORDS = ("cov", "impedance", "f0_hz", "H_m", "width_m", "lambda_over_w_0p1",
          "lambda_over_w_0p4", "r_h_over_w", "zeta_mean", "f0_H_over_vs")


def _from_practice() -> pd.DataFrame:
    rows = []
    root = AUDIT.parent
    for family, path in (("iid", root / "practice_iid_record.csv"), ("ood_dipping", root / "practice_ood_dipping_record.csv")):
        if not path.is_file():
            rows.append({"family": family, "status": "missing", "note": f"{path} missing"})
            continue
        frame = pd.read_csv(path)
        part = frame[frame["method"] == "P128n128"].copy()
        part["family"] = family
        part["status"] = "run_g0"
        part["source"] = path.name
        rows.append(part)
    if not rows:
        return pd.DataFrame()
    frames = [r for r in rows if isinstance(r, pd.DataFrame)]
    notes = [r for r in rows if isinstance(r, dict)]
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if notes:
        out = pd.concat([out, pd.DataFrame(notes)], ignore_index=True)
    return out


def _forward_missing(have: set[str]) -> pd.DataFrame:
    from data import IID_MESH_M, KIND_RUN, KIND_SECTION, Cache, make_window
    from diag_forward import load_fp32

    device = torch.device("cpu")
    model, stamp = load_fp32(CKPT, device)
    model.eval()
    cache = Cache(REPO / "data" / "vardomain" / "cache")
    freq = np.asarray(cache.freq, dtype=np.float64)
    frames = []
    if "ood_three_layer" not in have:
        frames.append(_forward_runs(model, cache, "ood_three_layer", device, freq, stamp))
    if "manifold" not in have:
        frames.append(_forward_sections(model, cache, device, freq, stamp))
    frames = [f for f in frames if f is not None and len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _batches(items, size):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _forward_runs(model, cache, family, device, freq, stamp) -> pd.DataFrame:
    from data import IID_MESH_M, KIND_RUN, make_window

    jobs = []
    for flat, (ci, j) in enumerate(cache.run_of):
        camp = cache.campaigns[ci]
        if camp.name != family or str(camp.split[j]) != "test" or not bool(camp.valid[j]):
            continue
        jobs.append(flat)
    curves_pred, curves_true, curves_prior, sids, stations, clusters, xs = [], [], [], [], [], [], []
    width = 500
    start = int(IID_MESH_M // 2 - width // 2)
    for chunk in _batches(jobs, 4):
        windows = [make_window(cache, "metric", KIND_RUN, flat, width, start, 0, False) for flat in chunk]
        x = torch.stack([w[0] for w in windows])
        query = torch.stack([w[1] for w in windows])
        with torch.inference_mode():
            g = model(x.to(device), query.to(device)).float().cpu().numpy()
        for local, flat in enumerate(chunk):
            prior = windows[local][3].numpy()
            true = windows[local][2].numpy()
            valid = windows[local][4].numpy().astype(bool)
            xabs = windows[local][6].numpy()
            keep = np.flatnonzero(valid)
            for station in keep:
                curves_pred.append(prior[station] + g[local, station])
                curves_true.append(true[station])
                curves_prior.append(prior[station])
                sids.append(f"{family}:{flat}")
                stations.append(int(station))
                clusters.append(str(int(cache.campaign(flat)[0].group[cache.campaign(flat)[1]])))
                xs.append(float(xabs[station]))
    if not sids:
        return pd.DataFrame([{"family": family, "status": "no_rows", "note": "no valid test rows"}])
    long = score_curves(
        sample_id=np.array(sids), station=np.array(stations), cluster_id=np.array(clusters),
        freq=freq, log_pred=np.stack(curves_pred), log_true=np.stack(curves_true),
        log_prior=np.stack(curves_prior), method="P128n128", x_m=np.array(xs),
    )
    long.to_csv(AUDIT / f"errors_{family}_long.csv", index=False)
    headline = aggregate(long, "record").frame
    headline["family"] = family
    headline["status"] = "scored_fp32"
    headline["epoch"] = stamp.get("epoch")
    headline["harm_rate_overall"] = harm_rate(long, "overall", "P128n128")
    return headline


def _forward_sections(model, cache, device, freq, stamp) -> pd.DataFrame:
    from data import KIND_SECTION, make_window

    curves_pred, curves_true, curves_prior, sids, stations, clusters, xs = [], [], [], [], [], [], []
    for index, sec in enumerate(cache.sections):
        if sec.split not in ("val", "blind", "test"):
            continue
        width = min(int(sec.lx), 2000)
        window = make_window(cache, "metric", KIND_SECTION, index, width, 0, 20, False)
        with torch.inference_mode():
            g = model(window[0].unsqueeze(0).to(device), window[1].unsqueeze(0).to(device)).float().cpu().numpy()[0]
        prior = window[3].numpy()
        true = window[2].numpy()
        valid = window[4].numpy().astype(bool)
        xabs = window[6].numpy()
        sid = sec.name.split("__")[-1]
        for station in np.flatnonzero(valid):
            curves_pred.append(prior[station] + g[station])
            curves_true.append(true[station])
            curves_prior.append(prior[station])
            sids.append(sid)
            stations.append(int(station))
            clusters.append(str(sec.group_id))
            xs.append(float(xabs[station]))
        if index % 10 == 0:
            print(f"manifold forward {index}", flush=True)
    if not sids:
        return pd.DataFrame([{"family": "manifold", "status": "no_rows"}])
    long = score_curves(
        sample_id=np.array(sids), station=np.array(stations), cluster_id=np.array(clusters),
        freq=freq, log_pred=np.stack(curves_pred), log_true=np.stack(curves_true),
        log_prior=np.stack(curves_prior), method="P128n128", x_m=np.array(xs),
    )
    long.to_csv(AUDIT / "errors_manifold_long.csv", index=False)
    headline = aggregate(long, "record").frame
    headline["family"] = "manifold"
    headline["status"] = "scored_fp32"
    headline["epoch"] = stamp.get("epoch")
    headline["harm_rate_overall"] = harm_rate(long, "overall", "P128n128")
    return headline


def _treasure() -> pd.DataFrame:
    path = AUDIT.parent / "treasure_long.csv"
    if not path.is_file():
        return pd.DataFrame([{"family": "treasure_island", "status": "missing", "note": "treasure_long.csv missing"}])
    long = pd.read_csv(path)
    part = long[long["method"] == "P128n128"]
    headline = aggregate(part, "record").frame
    headline["family"] = "treasure_island"
    headline["sample_id"] = "treasure_island"
    headline["status"] = "run_g0"
    return headline


def _spearman_and_bins(errors: pd.DataFrame, coverage: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    joined = errors.merge(coverage, on="sample_id", how="left", suffixes=("", "_cov"))
    spearman_rows = []
    bin_rows = []
    worst = []
    family_col = "family" if "family" in joined.columns else "family_cov"
    for family, part in joined.groupby(joined["family"].fillna(joined.get("family_cov"))):
        for band, band_part in part.groupby("band"):
            if "g_log_l1" not in band_part.columns or band_part["g_log_l1"].notna().sum() < 8:
                continue
            y = band_part["g_log_l1"]
            for coord in COORDS:
                if coord not in band_part.columns:
                    continue
                x = pd.to_numeric(band_part[coord], errors="coerce")
                mask = x.notna() & y.notna()
                if mask.sum() < 8:
                    continue
                rho = float(x[mask].corr(y[mask], method="spearman"))
                spearman_rows.append({
                    "table": "spearman", "family": family, "band": band, "coordinate": coord,
                    "spearman": rho, "n": int(mask.sum()), "status": "measured",
                })
                if band == "overall" and int(mask.sum()) >= 20:
                    ranks = pd.qcut(x[mask], q=5, duplicates="drop")
                    for label, bucket in y[mask].groupby(ranks, observed=True):
                        bin_rows.append({
                            "table": "partial", "family": family, "band": band, "coordinate": coord,
                            "bin": str(label), "mean_g_log_l1": float(bucket.mean()), "n": int(bucket.size),
                        })
        overall = part[part["band"] == "overall"].dropna(subset=["g_log_l1"])
        if len(overall):
            top = overall.nlargest(min(20, len(overall)), "g_log_l1")
            for rank, (_, row) in enumerate(top.iterrows(), start=1):
                worst.append({
                    "table": "worst", "family": family, "band": "overall", "rank": rank,
                    "sample_id": row["sample_id"], "g_log_l1": row["g_log_l1"],
                    "prior_log_l1": row.get("prior_log_l1", np.nan),
                    **{coord: row.get(coord, np.nan) for coord in COORDS},
                })
    return pd.DataFrame(spearman_rows + bin_rows + worst), joined


def plot(frame: pd.DataFrame, path) -> None:
    use_agg()
    import matplotlib.pyplot as plt

    part = frame[(frame["table"] == "partial") & (frame["family"] == "iid")]
    coords = [c for c in COORDS if c in set(part["coordinate"])][:6]
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.6))
    for ax, coord in zip(axes.ravel(), coords):
        hit = part[part["coordinate"] == coord]
        ax.plot(np.arange(len(hit)), hit["mean_g_log_l1"].to_numpy(), marker="o")
        ax.set_xticks(np.arange(len(hit)), [str(i + 1) for i in range(len(hit))])
        ax.set_title(coord, fontsize=9)
        ax.set_xlabel("quantile bin")
        ax.set_ylabel("mean g log-L1")
    fig.suptitle("IID overall error, partial dependence")
    fig.tight_layout()
    save_fig(fig, path)
    plt.close(fig)


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    practice = _from_practice()
    have = set(practice["family"].dropna().astype(str)) if "family" in practice.columns else set()
    print(f"practice families {sorted(have)}", flush=True)
    forwarded = _forward_missing(have)
    treasure = _treasure()
    errors = pd.concat([practice, forwarded, treasure], ignore_index=True)
    errors.to_csv(AUDIT / "errors_records.csv", index=False)
    coverage = pd.read_csv(AUDIT / "coverage.csv") if (AUDIT / "coverage.csv").is_file() else pd.DataFrame()
    usable = errors[errors["method"].eq("P128n128")] if "method" in errors.columns else errors.iloc[0:0]
    summary, _joined = _spearman_and_bins(usable, coverage) if len(coverage) else (pd.DataFrame(), usable)
    # Median error with a record bootstrap, one row per family and band.
    boot_rows = []
    if "g_log_l1" in usable.columns:
        for (family, band), part in usable.groupby(["family", "band"]):
            values = part["g_log_l1"].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size < 4 or family == "treasure_island":
                boot_rows.append({
                    "table": "error_median", "family": family, "band": band,
                    "median": float(np.median(values)) if values.size else np.nan,
                    "n": int(values.size), "status": "measured" if values.size else "empty",
                })
                continue
            boot = bootstrap_independent(
                values, part["sample_id"].astype(str).to_numpy(), unit="record",
                statistic=np.median, n_draw=400, rng=np.random.default_rng(0),
            )
            boot_rows.append({
                "table": "error_median", "family": family, "band": band,
                "median": boot["point"], "low": boot["low"], "high": boot["high"],
                "n": int(values.size), "status": "measured",
            })
    frame = pd.concat([summary, pd.DataFrame(boot_rows)], ignore_index=True)
    frame.to_csv(AUDIT / "errors.csv", index=False)
    if len(frame):
        plot(frame, AUDIT / "errors.png")
    provenance(
        AUDIT / "errors.provenance.json",
        campaign="iid and dipping center stations from run_g0; three-layer and manifold fp32 forward; treasure_long",
        engine="stored log_tf1d",
        outcrop="stored log_tf1d; Treasure Island uses the Phase 0 sidecar's outcrop",
        damping="qvs for IID, dipping, and three-layer; manifold as stored",
        fft=1310.72,
        checkpoint=str(CKPT),
        epoch=46,
        dtype="float32",
        unit="record",
        sample_ids=usable["sample_id"].astype(str).unique().tolist() if "sample_id" in usable.columns else ["none"],
        difference=(
            "IID and dipping errors are the run_g0 center-station scores. "
            "Three-layer is a centered 500 m window. Manifold sections use their own interior, stride 20 m. "
            "Only load_fp32 was loaded. Spearman is within a family. Partial dependence is the mean error in quantile bins."
        ),
    )
    print(f"error rows {len(frame)}", flush=True)


if __name__ == "__main__":
    main()
