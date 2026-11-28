#!/usr/bin/env python3
"""Score a vardomain model by window width, and compare runs.

    python evaluate.py --model-dir ../../checkpoints/vardomain/metric_s0 --cache ../../data/vardomain/cache
    python evaluate.py --summarize ../../checkpoints/vardomain

Metrics per labeled station, then averaged per (source, width) bin:
  log_l1      mean_f |log pred - log true|
  rel_l2      ||pred - true|| / ||true|| on linear |TF|
  pearson     Pearson of linear |TF| over 0.1-10 Hz
  pearson_tr  same over the trusted band 0.1-2.5 Hz
Consistency: a station at least CONSISTENCY_MARGIN_M from both edges of every
window that contains it should get one answer. ``consistency`` is the mean over
such stations of the standard deviation of log pred across those windows
(widths and positions), averaged over frequency.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import FAMILY_NAME, Cache, WindowSet, _loader_threads, eval_records
from model import VarDomainLOGLO, load_vardomain_state, model_from_config
from provenance import bind_training_autocast, full_stamp

TRUSTED_HZ = (0.1, 2.5)
CONSISTENCY_MARGIN_M = 100
BANDS = {"low": (0.1, 0.4), "mid": (0.4, 2.5), "high": (2.5, 10.0)}


def _pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    return (a * b).sum(-1) / np.maximum(np.sqrt((a * a).sum(-1) * (b * b).sum(-1)), 1e-12)


def _spearman(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ra = np.argsort(np.argsort(a, axis=-1), axis=-1).astype(np.float64)
    rb = np.argsort(np.argsort(b, axis=-1), axis=-1).astype(np.float64)
    return _pearson(ra, rb)


def _ccc(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    mx, my = x.mean(axis=-1), y.mean(axis=-1)
    vx, vy = x.var(axis=-1), y.var(axis=-1)
    cov = ((x - mx[:, None]) * (y - my[:, None])).mean(axis=-1)
    return 2.0 * cov / (vx + vy + (mx - my) ** 2 + 1e-12)


def _anderson(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Two-sample Anderson-Darling statistic, one value per station, on 48 frequencies."""
    import warnings

    from scipy.stats import anderson_ksamp

    step = max(int(pred.shape[-1] // 48), 1)
    out = np.empty(pred.shape[0], dtype=np.float64)
    for i in range(pred.shape[0]):
        left, right = pred[i, ::step], true[i, ::step]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out[i] = float(anderson_ksamp([left, right]).statistic)
        except ValueError:
            out[i] = np.nan
    return out


def _mode_scores(pred: np.ndarray, true: np.ndarray, freq: np.ndarray) -> dict[str, np.ndarray]:
    """Relative peak-frequency error, peak log-amplitude error, and ±25% window log-L1."""
    n = pred.shape[0]
    f = np.asarray(freq, dtype=np.float64)
    log_p, log_t = np.log(np.maximum(pred, 1e-12)), np.log(np.maximum(true, 1e-12))
    keys = (
        "m1_freq_err", "m2_freq_err", "m1_amp_log", "m2_amp_log", "m1_log_l1", "m2_log_l1",
        "m1_rel_l1", "m2_rel_l1", "m1_rel_lin", "m2_rel_lin",
    )
    out = {key: np.full(n, np.nan) for key in keys}
    low = (f >= 0.15) & (f <= 4.0)
    low_idx = np.flatnonzero(low)
    if low_idx.size == 0:
        return out
    for i in range(n):
        i1 = int(low_idx[np.argmax(true[i, low_idx])])
        f1 = float(f[i1])
        hi = np.flatnonzero((f >= 1.5 * f1) & (f <= 10.0))
        i2 = int(hi[np.argmax(true[i, hi])]) if hi.size else i1
        for tag, centre_i in (("m1", i1), ("m2", i2)):
            centre = float(f[centre_i])
            # Linear relative error at the true peak bin. m1_amp_log is a different
            # quantity: |log| at the predicted peak inside ±25% of this frequency.
            out[f"{tag}_rel_lin"][i] = abs(float(pred[i, centre_i]) - float(true[i, centre_i])) / max(float(true[i, centre_i]), 1e-12)
            band = (f >= 0.75 * centre) & (f <= 1.25 * centre)
            if not np.any(band):
                continue
            j = int(np.flatnonzero(band)[np.argmax(pred[i, band])])
            out[f"{tag}_freq_err"][i] = abs(float(f[j]) - centre) / max(centre, 1e-6)
            out[f"{tag}_amp_log"][i] = abs(float(log_p[i, j]) - float(log_t[i, centre_i]))
            out[f"{tag}_log_l1"][i] = float(np.mean(np.abs(log_p[i, band] - log_t[i, band])))
            out[f"{tag}_rel_l1"][i] = float(np.sum(np.abs(pred[i, band] - true[i, band])) / max(np.sum(true[i, band]), 1e-12))
    return out


def station_metrics(
    log_pred: np.ndarray, log_true: np.ndarray, freq: np.ndarray, log_prior: np.ndarray | None = None, *,
    core: bool = False, sqrt_overall: bool = False,
) -> dict[str, np.ndarray]:
    """``core`` keeps the log-L1 the selection score uses and skips the per-station loops."""
    p, t = np.exp(log_pred.astype(np.float64)), np.exp(log_true.astype(np.float64))
    tr = (freq >= TRUSTED_HZ[0]) & (freq <= TRUSTED_HZ[1])
    la, lb = log_pred.astype(np.float64), log_true.astype(np.float64)
    out = {
        "log_l1": np.abs(la - lb).mean(-1),
        "rel_l1": np.abs(p - t).sum(-1) / np.maximum(t.sum(-1), 1e-12),
        "rel_l2": np.linalg.norm(p - t, axis=-1) / np.maximum(np.linalg.norm(t, axis=-1), 1e-12),
        "pearson": _pearson(p, t),
        "pearson_log": _pearson(la, lb),
        "pearson_tr": _pearson(p[:, tr], t[:, tr]),
        "ccc_log": _ccc(la, lb),
    }
    if not core:
        out["spearman"] = _spearman(p, t)
        out["anderson"] = _anderson(p, t)
        out.update(_mode_scores(p, t, freq))
    for name, (lo, hi) in BANDS.items():
        band = (freq >= lo) & (freq <= hi)
        out[f"log_l1_{name}"] = np.abs(log_pred[:, band] - log_true[:, band]).mean(-1)
        if log_prior is not None:
            base = np.abs(log_prior[:, band] - log_true[:, band]).mean(-1)
            out[f"skill_{name}"] = 1.0 - out[f"log_l1_{name}"] / np.maximum(base, 1e-6)
    if log_prior is not None:
        base_full = np.abs(log_prior.astype(np.float64) - log_true.astype(np.float64)).mean(-1)
        out["skill_full"] = 1.0 - out["log_l1"] / np.maximum(base_full, 1e-6)
    if sqrt_overall:
        # Opt-in. The default call does not add this key, and selection_score
        # does not read it. |log pred − log true| is |g_pred − g_true|.
        from analysis.score_harness import weighted_overall_g_log_l1

        out["log_l1_overall_sqrt"] = np.asarray(weighted_overall_g_log_l1(la, lb, freq), dtype=np.float64)
    return out


@torch.no_grad()
def predict(model, loader, device, amp: bool, target: str):
    """Yield (log_pred, log_true, log_prior, info, x_abs, x_local) per labeled station."""
    model.eval()
    for x, query, tf, prior, valid, info, x_abs in loader:
        x = x.to(device, non_blocking=True)
        query = query.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            head = getattr(model, "deltaphi", None)
            index = getattr(model, "_phi_index", None)
            if head is not None and index is not None:
                from deltaphi import retrieve_batch
                from model import sample_grid

                coeff, latent, _recon = model.encode(x)
                raw = model.readout(coeff, latent, query)
                g_k, sim = retrieve_batch(index, x, query, raw, info, train=False)
                raw = g_k + head(sample_grid(latent, query), g_k, sim)
            else:
                raw = model(x, query)
            raw = raw.float()
        pred = (raw + prior.to(device)).cpu().numpy() if target == "residual" else raw.cpu().numpy()
        v = valid.numpy()
        b, k = np.nonzero(v)
        inf = info.numpy()[b]
        xa = x_abs.numpy()[b, k]
        yield pred[b, k], tf.numpy()[b, k], prior.numpy()[b, k], inf, xa, xa - inf[:, 3]


def evaluate(model, cache: Cache, encoding: str, split: str, device, *, batch: int = 8, workers: int = 4,
             iid_limit: int | None = None, positions: int = 5, amp: bool = True, target: str = "residual",
             families: tuple[str, ...] = ("iid", "ood_dipping", "ood_three_layer"), stage: int = 0,
             extra_section_splits: tuple[str, ...] = (),
             section_group_ids: set[str] | None = None, core: bool = False,
             sqrt_overall: bool = False) -> dict:
    records = eval_records(
        cache, split, iid_limit=iid_limit, positions=positions, families=families,
        section_stride=5, widths=(500, 1000, 1500, 2000) if positions > 1 else (500,),
        extra_section_splits=extra_section_splits, section_group_ids=section_group_ids,
    )
    if not records:
        return {"split": split, "encoding": encoding, "bins": {}}
    loader_kwargs = {
        "batch_size": batch, "shuffle": False, "num_workers": workers, "pin_memory": device.type == "cuda",
    }
    if workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["worker_init_fn"] = _loader_threads
    loader = DataLoader(WindowSet(cache, encoding, records, stage, seed=0, mirror=False), **loader_kwargs)
    bins: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    groups: dict[tuple, list] = defaultdict(list)
    for log_pred, log_true, log_prior, info, x_abs, x_local in predict(model, loader, device, amp, target):
        m = station_metrics(log_pred, log_true, cache.freq, log_prior, core=core, sqrt_overall=sqrt_overall)
        for i in range(log_pred.shape[0]):
            fam = FAMILY_NAME.get(int(info[i, 4]), "section")
            width = int(info[i, 2])
            flags = int(info[i, 6])
            near = bool(flags & 1)
            blind = bool(flags & 2)
            if fam == "manifold" and blind:
                keys = [f"manifold_blind_w{width}", f"manifold_blind_{'near' if near else 'lateral'}_w{width}"]
            elif fam == "manifold":
                keys = [f"manifold_w{width}", f"manifold_{'near' if near else 'lateral'}_w{width}"]
            else:
                keys = [f"{fam}_w{width}"]
            for key in keys:
                for name, vals in m.items():
                    bins[key][name].append(float(vals[i]))
            if CONSISTENCY_MARGIN_M <= x_local[i] <= width - CONSISTENCY_MARGIN_M:
                groups[(int(info[i, 4]), int(info[i, 1]), round(float(x_abs[i]), 1))].append(log_pred[i].astype(np.float32))
    out: dict = {"split": split, "encoding": encoding, "target": target, "bins": {}}
    for key in sorted(bins, key=lambda s: (s.split("_w")[0], int(s.split("_w")[1]))):
        out["bins"][key] = {name: float(np.nanmean(v)) for name, v in bins[key].items()} | {"n_stations": len(bins[key]["log_l1"])}
    for src in (
        "iid", "ood_dipping", "ood_three_layer", "manifold",
        "manifold_near", "manifold_lateral", "manifold_blind",
        "manifold_blind_near", "manifold_blind_lateral",
    ):
        keys = [k for k in out["bins"] if k.startswith(src + "_w")]
        if keys:
            names = (
                "log_l1", "rel_l1", "rel_l2", "pearson", "pearson_log", "pearson_tr", "spearman", "ccc_log",
                "anderson", "log_l1_low", "log_l1_mid", "log_l1_high", "skill_full", "skill_mid",
                "m1_freq_err", "m2_freq_err", "m1_amp_log", "m2_amp_log", "m1_log_l1", "m2_log_l1",
                "m1_rel_lin", "m2_rel_lin", "log_l1_overall_sqrt",
            )
            out[f"{src}_mean"] = {
                n: float(np.mean(vals))
                for n in names
                if (vals := [out["bins"][k][n] for k in keys if n in out["bins"][k]])
            }
    stds = [np.std(np.stack(v), axis=0).mean() for v in groups.values() if len(v) > 1]
    if stds:
        out["consistency"] = float(np.mean(stds))
        out["consistency_n"] = len(stds)
    return out


@torch.no_grad()
def seed_variance(model, cache: Cache, device, *, target: str, split: str = "test", amp: bool = True) -> dict:
    """Within-seed sigma ratio and the share of seed-level variance explained, on IID."""
    records = [r for r in eval_records(cache, split, families=("iid",), positions=1) if int(r["width"]) == 500]
    if not records:
        return {}
    by_run: dict[tuple, list] = defaultdict(list)
    loader = DataLoader(WindowSet(cache, "metric", records, 0, seed=0, mirror=False), batch_size=4, shuffle=False)
    for log_pred, log_true, _prior, info, _xa, _xl in predict(model, loader, device, amp, target):
        for i in range(log_pred.shape[0]):
            by_run[(int(info[i, 5]), int(info[i, 1]))].append((log_pred[i], log_true[i]))
    # each run should contribute 21 stations in order
    by_group: dict[int, list] = defaultdict(list)
    for (group, _run), pairs in by_run.items():
        by_group[group].append((np.stack([p[0] for p in pairs]), np.stack([t[1] for t in pairs])))
    band = (cache.freq >= 0.4) & (cache.freq <= 2.5)
    num = den = sig_p = sig_t = 0.0
    n_groups = 0
    for stacks in by_group.values():
        if len(stacks) < 2:
            continue
        pr = np.stack([p[:, band] for p, _ in stacks])  # seed, station, freq
        tr = np.stack([t[:, band] for _, t in stacks])
        sig_p += float(pr.std(axis=0).mean())
        sig_t += float(tr.std(axis=0).mean())
        dp, dt = pr - pr.mean(0), tr - tr.mean(0)
        num += float(((dp - dt) ** 2).sum())
        den += float((dt ** 2).sum())
        n_groups += 1
    if n_groups == 0:
        return {"n_groups": 0}
    return {
        "n_groups": n_groups,
        "sigma_ratio_mid": sig_p / max(sig_t, 1e-8),
        "seed_explained_mid": 1.0 - num / max(den, 1e-8),
    }


def load_model(model_dir: Path, device) -> tuple[VarDomainLOGLO, dict]:
    """Load a checkpoint and bind the autocast training used.

    On CUDA, ``config["amp"]`` true means bfloat16 autocast. A float32 call of
    the same weights is a different map. The returned config is the checkpoint
    config; the bound forward is what scoring should call.
    """
    ck = torch.load(model_dir / "best_model.pt", map_location="cpu", weights_only=False)
    cfg = ck["config"]
    pca_mean = pca_components = None
    if cfg.get("depth_pca"):
        pca_mean = np.load(model_dir / "pca_mean.npy")
        pca_components = np.load(model_dir / "pca_components.npy")
    model = model_from_config(
        cfg, np.load(model_dir / "pod_modes.npy"), np.load(model_dir / "pod_mean.npy"), pca_mean, pca_components,
    )
    load_vardomain_state(model, ck["state_dict"])
    model = model.to(device)
    bind_training_autocast(model, device, bool(cfg.get("amp", True)))
    return model, cfg


def summarize(root: Path) -> None:
    rows = []
    for path in sorted(root.glob("*/eval_test.json")):
        d = json.loads(path.read_text())
        row = {"run": path.parent.name, "encoding": d["encoding"]}
        for key, m in d["bins"].items():
            row[f"{key}_log_l1"] = m["log_l1"]
            row[f"{key}_pearson_tr"] = m["pearson_tr"]
        for k in ("section_consistency", "iid_consistency"):
            row[k] = d.get(k)
        rows.append(row)
    if not rows:
        raise SystemExit(f"no */eval_test.json under {root}")
    keys = sorted({k for r in rows for k in r} - {"run", "encoding"})
    by_enc = defaultdict(list)
    for r in rows:
        by_enc[r["encoding"]].append(r)
    summary = {enc: {k: [float(np.mean([r[k] for r in rs if r.get(k) is not None])), float(np.std([r[k] for r in rs if r.get(k) is not None]))]
                     for k in keys if any(r.get(k) is not None for r in rs)} | {"runs": [r["run"] for r in rs]}
               for enc, rs in by_enc.items()}
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"{'metric':38s}" + "".join(f"{enc:>24s}" for enc in summary))
    for k in keys:
        cells = "".join(f"{summary[e][k][0]:>15.4f} +- {summary[e][k][1]:.4f}" if k in summary[e] else f"{'':>24s}" for e in summary)
        print(f"{k:38s}{cells}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path)
    ap.add_argument("--cache", type=Path)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--iid-limit", type=int, default=None)
    ap.add_argument("--summarize", type=Path, help="aggregate */eval_test.json under this directory by encoding")
    args = ap.parse_args()
    if args.summarize:
        summarize(args.summarize)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.model_dir, device)
    res = evaluate(
        model, Cache(args.cache), cfg["encoding"], args.split, device, batch=args.batch,
        workers=args.workers, iid_limit=args.iid_limit, amp=cfg.get("amp", True),
        target=cfg.get("target", "residual"),
    )
    res["provenance"] = full_stamp(args.model_dir, device, bool(cfg.get("amp", True)))
    (args.model_dir / f"eval_{args.split}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
