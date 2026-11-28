#!/usr/bin/env python3
"""Phase 1.7 split integrity, the epoch-46 rule, and adversarial shift."""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import AUDIT, BOX, CACHE, CKPT, REPO, provenance, save_fig, use_agg
from build_cache import manifold_holdout, split_by_group

SCALAR_COLUMNS = ("cov", "impedance", "f0_hz", "H_m", "width_m", "zeta_mean")


def _split_checks() -> list[dict]:
    rows = []
    for family in ("iid", "ood_dipping", "ood_three_layer"):
        group = np.load(CACHE / family / "group.npy")
        stored = np.load(CACHE / family / "split.npy").astype(str)
        fresh = split_by_group(group, seed=42).astype(str)
        mismatch = int(np.sum(stored != fresh))
        crossed = 0
        for gid in np.unique(group):
            labels = set(stored[group == gid].tolist())
            if len(labels) > 1:
                crossed += 1
        rows.append({
            "table": "split_identity",
            "family": family,
            "n_records": int(stored.size),
            "n_groups": int(np.unique(group).size),
            "mismatch_vs_split_by_group": mismatch,
            "groups_crossing_splits": crossed,
            "status": "pass" if mismatch == 0 and crossed == 0 else "fail",
            "note": "split_by_group(group, seed=42) against cache split.npy. A group is one Sobol id.",
        })
    manifest = BOX / "vardomain" / "manifold150" / "manifest.csv"
    if not manifest.is_file():
        rows.append({
            "table": "split_identity", "family": "manifold", "status": "unread",
            "note": f"{manifest} missing",
        })
        return rows
    frame = pd.read_csv(manifest)
    complete = frame[frame["status"].astype(str) == "complete"] if "status" in frame.columns else frame
    assigned = manifold_holdout(complete.to_dict(orient="records"), seed=42)
    cache_split = {}
    for path in (CACHE / "sections").glob("*.npz"):
        sid = path.stem.split("__")[-1]
        cache_split[sid] = str(np.load(path)["split"])
    compared = 0
    mismatch = 0
    missing_cache = 0
    for sid, lab in assigned.items():
        if sid not in cache_split:
            missing_cache += 1
            continue
        compared += 1
        mismatch += int(cache_split[sid] != lab)
    extra = sorted(set(cache_split) - set(assigned))
    group_cross = 0
    joined = complete.copy()
    joined["holdout"] = joined["sample_id"].map(assigned)
    for _, part in joined.groupby("group_id"):
        labels = set(part["holdout"].dropna().astype(str))
        if len(labels) > 1:
            group_cross += 1
    seed_cross = 0
    if "field_seed" in joined.columns:
        for _, part in joined.groupby("field_seed"):
            labels = set(part["holdout"].dropna().astype(str))
            if len(labels) > 1:
                seed_cross += 1
    rows.append({
        "table": "split_identity",
        "family": "manifold",
        "n_manifest_complete": int(len(complete)),
        "n_cache": len(cache_split),
        "n_compared": compared,
        "mismatch_vs_manifold_holdout": mismatch,
        "missing_from_cache": missing_cache,
        "cache_ids_not_in_holdout": len(extra),
        "groups_crossing_splits": group_cross,
        "seeds_crossing_splits": seed_cross,
        "status": "pass" if mismatch == 0 and group_cross == 0 and seed_cross == 0 else "fail",
        "note": "manifold_holdout(seed=42) on complete manifest rows versus the section npz split. Blind stays blind.",
    })
    return rows


def _seed_rows() -> list[dict]:
    rows = []
    specs = (
        ("iid", BOX / "transfer_function" / "manifest.csv", "run_index", "rf_seed"),
        ("ood_dipping", BOX / "ood_dipping" / "manifest.csv", "index", "rf_seed"),
        ("ood_three_layer", BOX / "ood_three_layer" / "manifest.csv", "index", "seed1"),
    )
    for family, path, run_key, seed_key in specs:
        if not path.is_file():
            rows.append({"table": "seed_crossing", "family": family, "status": "unread", "note": f"{path} missing"})
            continue
        manifest = pd.read_csv(path)
        run_index = np.load(CACHE / family / "run_index.npy")
        split = np.load(CACHE / family / "split.npy").astype(str)
        group = np.load(CACHE / family / "group.npy")
        lookup = manifest.set_index(run_key)
        seeds = []
        missing = 0
        for rid in run_index:
            if int(rid) not in lookup.index:
                missing += 1
                seeds.append(np.nan)
            else:
                seeds.append(lookup.loc[int(rid), seed_key] if seed_key in lookup.columns else np.nan)
        seeds = np.asarray(seeds)
        crossed = 0
        values = pd.Series(seeds).astype(str)
        for seed, index in values.groupby(values).groups.items():
            if seed in ("nan", "None"):
                continue
            labels = set(split[np.asarray(index, dtype=int)].tolist())
            if len(labels) > 1:
                crossed += 1
        # A reused random-field seed across different Sobol groups is not a split leak by itself.
        # Count only seeds that land in more than one split.
        rows.append({
            "table": "seed_crossing",
            "family": family,
            "seed_column": seed_key,
            "n_missing_manifest_rows": missing,
            "seeds_crossing_splits": crossed,
            "groups_crossing_splits": int(sum(
                len(set(split[group == gid].tolist())) > 1 for gid in np.unique(group)
            )),
            "status": "pass" if crossed == 0 and missing == 0 else "fail",
            "note": "A seed crosses when the same manifest seed value appears in two cache splits.",
        })
    return rows


def _epoch_rows() -> list[dict]:
    ck = torch_load()
    cfg = ck.get("config", {})
    val = ck.get("val") if isinstance(ck.get("val"), dict) else {}
    blind_keys = [k for k in val if "blind" in str(k)]
    history = CKPT / "history.csv"
    return [{
        "table": "epoch_rule",
        "epoch": int(ck.get("epoch", -1)),
        "val_score": ck.get("val_score"),
        "iid_log_l1": ck.get("iid_log_l1"),
        "select_families": cfg.get("select_families"),
        "agree_floor": cfg.get("agree_floor"),
        "query_dtype": cfg.get("query_dtype"),
        "amp": cfg.get("amp"),
        "stage": cfg.get("stage"),
        "history_csv_present": history.is_file(),
        "blind_keys_in_saved_val": ",".join(blind_keys),
        "status": "recorded",
        "note": (
            "train.selection_score for the unified stage is manifold validation log-L1 only, and blind is logged rather than selected. "
            "This checkpoint's config select_families is the p128 branch: the mean of IID and manifold validation log-L1, "
            "with bf16 agreement under AGREE_FLOOR 0.0136 and query dtype float32. "
            "Blind is not in select_families. history.csv is not in the checkpoint directory, so the claim that epoch 46 "
            "is the minimum of the curve was not recomputed."
        ),
    }]


def torch_load():
    import torch
    return torch.load(CKPT / "best_model.pt", map_location="cpu", weights_only=False)


def _features() -> pd.DataFrame:
    pca = pd.read_csv(AUDIT / "pca_records.csv")
    codes = np.load(AUDIT / "pca_codes.npy")
    index = pd.read_csv(AUDIT / "pca_codes_index.csv")
    if len(index) != codes.shape[0]:
        raise ValueError("pca code index does not match pca_codes.npy")
    order = index["sample_id"].astype(str)
    pca = pca.set_index(pca["sample_id"].astype(str)).loc[order].reset_index(drop=True)
    for i in range(16):
        pca[f"c{i}"] = codes[:, i]
    coverage = pd.read_csv(AUDIT / "coverage.csv")
    keep = ["sample_id", *SCALAR_COLUMNS]
    keep = [c for c in keep if c in coverage.columns]
    merged = pca.merge(coverage[keep], on="sample_id", how="left")
    return merged


def _auc(frame: pd.DataFrame, mask_a, mask_b, name: str) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    left = frame[mask_a]
    right = frame[mask_b]
    cols = [f"c{i}" for i in range(16) if f"c{i}" in frame.columns]
    cols += [c for c in SCALAR_COLUMNS if c in frame.columns]
    part = pd.concat([left, right], ignore_index=True)
    x = part[cols].apply(pd.to_numeric, errors="coerce")
    good = np.isfinite(x.to_numpy(dtype=float)).all(axis=1)
    x = x.loc[good]
    y = np.array([0] * len(left) + [1] * len(right))[np.asarray(good)]
    row = {
        "table": "adversarial", "comparison": name, "n_a": int((y == 0).sum()), "n_b": int((y == 1).sum()),
        "n_features": len(cols),
    }
    if y.size < 20 or len(np.unique(y)) < 2 or min(np.bincount(y.astype(int))) < 5:
        row.update({"status": "too_small", "auc_mean": np.nan, "note": "Not enough rows for a 5-fold logistic."})
        return row
    folds = int(min(5, np.bincount(y.astype(int)).min()))
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=400, C=0.5))
    try:
        scores = cross_val_score(model, x.to_numpy(dtype=float), y, cv=folds, scoring="roc_auc")
        row.update({
            "status": "measured",
            "auc_mean": float(np.mean(scores)),
            "auc_std": float(np.std(scores)),
            "cv": folds,
            "note": "Logistic regression on 16 depth-PCA coefficients plus coverage scalars. The seismic model was not trained.",
        })
    except Exception as exc:
        row.update({"status": "failed", "auc_mean": np.nan, "note": f"{type(exc).__name__}: {exc}"})
    return row


def _distances(frame: pd.DataFrame) -> pd.DataFrame:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    cols = [f"c{i}" for i in range(16) if f"c{i}" in frame.columns]
    cols += [c for c in SCALAR_COLUMNS if c in frame.columns]
    train = frame[(frame["family"] == "iid") & (frame["split"] == "train")].copy()
    x_train = train[cols].apply(pd.to_numeric, errors="coerce")
    good = np.isfinite(x_train.to_numpy(dtype=float)).all(axis=1)
    scaler = StandardScaler().fit(x_train.loc[good].to_numpy(dtype=float))
    neighbors = NearestNeighbors(n_neighbors=1).fit(scaler.transform(x_train.loc[good].to_numpy(dtype=float)))
    query = frame[cols].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(query.to_numpy(dtype=float)).all(axis=1)
    dist = np.full(len(frame), np.nan)
    if finite.any():
        transformed = scaler.transform(query.loc[finite].to_numpy(dtype=float))
        dist[np.flatnonzero(np.asarray(finite))] = neighbors.kneighbors(transformed)[0][:, 0]
    out = frame[["sample_id", "family", "split"]].copy()
    out["nn_distance_to_iid_train"] = dist
    out["table"] = "nearest_neighbor"
    return out


def _error_join(dist: pd.DataFrame) -> list[dict]:
    """Spearman of distance and the center-station error already written by run_g0."""

    rows = []
    practice = AUDIT.parent
    pieces = []
    for family, path in (("iid", practice / "practice_iid_record.csv"), ("ood_dipping", practice / "practice_ood_dipping_record.csv")):
        if not path.is_file():
            rows.append({"table": "distance_vs_error", "family": family, "status": "missing", "note": f"{path.name} missing"})
            continue
        score = pd.read_csv(path)
        score = score[(score["method"] == "P128n128") & (score["band"] == "overall")]
        score = score.rename(columns={"g_log_l1": "error"})
        pieces.append(score[["sample_id", "error"]].assign(family=family))
    if not pieces:
        return rows
    errors = pd.concat(pieces, ignore_index=True)
    joined = dist.merge(errors, on=["sample_id", "family"], how="inner")
    for family, part in joined.groupby("family"):
        if len(part) < 10:
            rows.append({"table": "distance_vs_error", "family": family, "status": "too_small", "n": int(len(part))})
            continue
        spearman = float(pd.Series(part["nn_distance_to_iid_train"]).corr(part["error"], method="spearman"))
        rows.append({
            "table": "distance_vs_error",
            "family": family,
            "band": "overall",
            "n": int(len(part)),
            "spearman": spearman,
            "status": "measured",
            "note": "Center-station overall g log-L1 from run_g0 practice tables, against nearest-neighbor distance in the PCA-plus-scalar space.",
        })
    ti = dist[dist["family"] == "treasure_island"]
    others = dist[dist["family"].isin(["iid", "ood_dipping", "ood_three_layer"]) & (dist["split"] == "test")]
    if len(ti) and len(others):
        value = float(ti["nn_distance_to_iid_train"].iloc[0])
        pool = others["nn_distance_to_iid_train"].to_numpy(dtype=float)
        pool = pool[np.isfinite(pool)]
        percentile = float(np.mean(pool <= value)) if pool.size else np.nan
        rows.append({
            "table": "treasure_distance",
            "nn_distance_to_iid_train": value,
            "percentile_among_synthetic_test": percentile,
            "n_test": int(pool.size),
            "status": "measured",
            "note": "Treasure Island's distance to the IID training cloud, as a percentile of synthetic test distances.",
        })
    return rows


def plot(frame: pd.DataFrame, dist: pd.DataFrame, path) -> None:
    use_agg()
    import matplotlib.pyplot as plt

    adv = frame[frame["table"] == "adversarial"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
    if len(adv):
        axes[0].barh(adv["comparison"].astype(str), adv["auc_mean"].to_numpy(dtype=float), color="C0")
        axes[0].axvline(0.5, color="0.4", ls="--", lw=1)
        axes[0].set_xlabel("cross-validated AUC")
        axes[0].set_title("Adversarial validation")
    if dist is not None and len(dist):
        for family, part in dist.groupby("family"):
            values = part["nn_distance_to_iid_train"].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            axes[1].hist(values, bins=30, histtype="step", density=True, label=family)
        ti = dist[dist["family"] == "treasure_island"]
        if len(ti) and np.isfinite(ti["nn_distance_to_iid_train"].iloc[0]):
            axes[1].axvline(float(ti["nn_distance_to_iid_train"].iloc[0]), color="C3", label="Treasure Island")
        axes[1].set_xlabel("distance to nearest IID training record")
        axes[1].legend(frameon=False, fontsize=7)
        axes[1].set_title("Shift in the depth-PCA space")
    fig.tight_layout()
    save_fig(fig, path)
    plt.close(fig)


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    rows = []
    rows.extend(_split_checks())
    rows.extend(_seed_rows())
    rows.extend(_epoch_rows())
    feature_note = ""
    dist = pd.DataFrame()
    try:
        features = _features()
    except Exception as exc:
        features = None
        feature_note = f"{type(exc).__name__}: {exc}"
        rows.append({"table": "adversarial", "comparison": "all", "status": "unread", "note": feature_note})
    if features is not None:
        iid_train = (features["family"] == "iid") & (features["split"] == "train")
        comparisons = (
            ("iid train vs iid test", iid_train, (features["family"] == "iid") & (features["split"] == "test")),
            ("iid train vs dipping test", iid_train, (features["family"] == "ood_dipping") & (features["split"] == "test")),
            ("iid train vs three-layer test", iid_train, (features["family"] == "ood_three_layer") & (features["split"] == "test")),
            ("iid train vs manifold", iid_train, features["family"] == "manifold"),
            ("manifold train vs manifold val", (features["family"] == "manifold") & (features["split"] == "train"),
             (features["family"] == "manifold") & (features["split"] == "val")),
            ("manifold train vs manifold blind", (features["family"] == "manifold") & (features["split"] == "train"),
             (features["family"] == "manifold") & (features["split"] == "blind")),
        )
        for name, a, b in comparisons:
            rows.append(_auc(features, a, b, name))
        dist = _distances(features)
        dist.to_csv(AUDIT / "split_distances.csv", index=False)
        rows.extend(_error_join(dist))
    frame = pd.DataFrame(rows)
    frame.to_csv(AUDIT / "splits.csv", index=False)
    plot(frame, dist, AUDIT / "splits.png")
    provenance(
        AUDIT / "splits.provenance.json",
        campaign="cache splits, manifold150 manifest, P128n128 config",
        engine="not a transfer-function score",
        outcrop="not used",
        damping="not used",
        fft=None,
        checkpoint=str(CKPT),
        epoch=46,
        dtype="float32 selection rule; adversarial classifier is a logistic on PCA coefficients",
        unit="record",
        sample_ids=frame["family"].dropna().astype(str).unique().tolist() if "family" in frame.columns else ["splits"],
        difference=(
            "Group leakage is a shared Sobol id or group_id across splits. "
            "Adversarial AUC is a small logistic on 16 PCA coefficients plus scalars, not a fine-tune of the seismic model. "
            "The stored checkpoint selects on iid,manifold, which is the p128 protocol, while selection_score() for a unified stage is manifold validation only. "
            "Blind is in neither score. "
            + feature_note
        ),
    )
    failed = frame[frame["status"] == "fail"] if "status" in frame.columns else frame.iloc[0:0]
    print(f"split failures {len(failed)}", flush=True)


if __name__ == "__main__":
    main()
