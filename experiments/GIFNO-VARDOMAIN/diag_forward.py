#!/usr/bin/env python3
"""Forward diagnostics on the current best checkpoints.

Float32, the same unweighted log-L1 the validation call reports. Train IID is
a fixed 24-group subsample; every manifold training section is scored. The
seed ensemble is the mean of log|TF| from P128 and P128s1. Mixed averages are
reported beside it. Each JSON names the checkpoint epoch it was scored on.

    python diag_forward.py --cache data/vardomain/cache
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "experiments" / "case7a"))

from data import STAGE_UNIFIED, FAMILY_NAME, Cache, WindowSet, eval_records  # noqa: E402
from encoding import encode, query_columns  # noqa: E402
from model import load_vardomain_state, model_from_config  # noqa: E402
from paper import p128n128_case7 as case7  # noqa: E402

ROOT = REPO / "checkpoints" / "vardomain" / "v2_ablations"
MODELS = ("P128", "P128s1", "P128n128", "P128c")
ENS = {
    "ens_seed": ("P128", "P128s1"),
    "ens_neck_contact": ("P128n128", "P128c"),
    "ens_all": MODELS,
}
PAIRS = (("P128", "P128s1"), ("P128", "P128c"), ("P128n128", "P128c"))
BANDS = (("full", 0.1, 10.0), ("low", 0.1, 0.4), ("mid", 0.4, 2.5), ("high", 2.5, 10.0))
OUT = HERE / "paper" / "diagnostics"
TRAIN_IID_GROUPS = 24


def load_fp32(model_dir: Path, device: torch.device):
    ck = torch.load(model_dir / "best_model.pt", map_location="cpu", weights_only=False)
    cfg = dict(ck["config"])
    pca_mean = pca_components = None
    if cfg.get("depth_pca"):
        pca_mean = np.load(model_dir / "pca_mean.npy")
        pca_components = np.load(model_dir / "pca_components.npy")
    if cfg.get("contact_features") and cfg.get("contact_mean") is None:
        cfg["contact_mean"] = np.load(model_dir / "contact_mean.npy").tolist()
        cfg["contact_std"] = np.load(model_dir / "contact_std.npy").tolist()
    model = model_from_config(
        cfg, np.load(model_dir / "pod_modes.npy"), np.load(model_dir / "pod_mean.npy"), pca_mean, pca_components,
    )
    load_vardomain_state(model, ck["state_dict"])
    model.to(device).eval()
    return model, {"epoch": int(ck.get("epoch", -1)), "val_score": ck.get("val_score"), "narrow": cfg.get("narrow"),
                   "contact_features": bool(cfg.get("contact_features")), "seed": cfg.get("seed")}


def _family(info_row: np.ndarray) -> str:
    fam = FAMILY_NAME.get(int(info_row[4]), "section")
    if fam != "manifold":
        return fam
    flags = int(info_row[6])
    near = bool(flags & 1)
    blind = bool(flags & 2)
    if blind:
        return "manifold_blind_near" if near else "manifold_blind_lateral"
    return "manifold_near" if near else "manifold_lateral"


def _limit_train(cache: Cache, records: list[dict]) -> list[dict]:
    """24 IID geology groups, every seed, plus every training section."""
    groups: list[int] = []
    seen: set[int] = set()
    for rec in records:
        if rec["kind"] != "run" or rec.get("family") != "iid":
            continue
        camp, j = cache.campaign(int(rec["index"]))
        group = int(camp.group[j])
        if group not in seen:
            seen.add(group)
            groups.append(group)
    rng = np.random.default_rng(0)
    chosen = set(int(g) for g in rng.choice(np.array(groups), size=min(TRAIN_IID_GROUPS, len(groups)), replace=False))
    kept = []
    for rec in records:
        if rec["kind"] == "section":
            kept.append(rec)
            continue
        if rec.get("family") != "iid":
            continue
        camp, j = cache.campaign(int(rec["index"]))
        if int(camp.group[j]) in chosen:
            kept.append(rec)
    return kept


class _Bucket:
    def __init__(self) -> None:
        self.sum: dict[tuple[str, str], float] = {}
        self.base: dict[tuple[str, str], float] = {}
        self.n: dict[tuple[str, str], int] = {}
        self.pair: dict[tuple[str, str, str], float] = {}
        self.pair_n: dict[tuple[str, str, str], int] = {}

    def add(self, model: str, family: str, band: str, width: int, err: float, prior: float, weight: int) -> None:
        key = (model, family, band, width)
        self.sum[key] = self.sum.get(key, 0.0) + err * weight
        self.base[key] = self.base.get(key, 0.0) + prior * weight
        self.n[key] = self.n.get(key, 0) + weight

    def add_pair(self, a: str, b: str, family: str, band: str, value: float, weight: int) -> None:
        key = (f"{a}-{b}", family, band)
        self.pair[key] = self.pair.get(key, 0.0) + value * weight
        self.pair_n[key] = self.pair_n.get(key, 0) + weight

    def table(self) -> dict:
        rows = []
        for (model, family, band, width), total in sorted(self.sum.items()):
            n = self.n[(model, family, band, width)]
            err = total / n
            base = self.base[(model, family, band, width)] / n
            rows.append({
                "model": model, "family": family, "band": band, "width": width, "n": n,
                "log_l1": err, "prior_log_l1": base, "skill": 1.0 - err / max(base, 1e-6),
            })
        pairs = []
        for key, total in sorted(self.pair.items()):
            n = self.pair_n[key]
            pairs.append({"pair": key[0], "family": key[1], "band": key[2], "n": n, "mean_abs_g": total / n})
        means = []
        grouped: dict[tuple[str, str, str], list[float]] = {}
        grouped_base: dict[tuple[str, str, str], list[float]] = {}
        for row in rows:
            key = (row["model"], row["family"], row["band"])
            grouped.setdefault(key, []).append(row["log_l1"])
            grouped_base.setdefault(key, []).append(row["prior_log_l1"])
        for key, vals in sorted(grouped.items()):
            err = float(np.mean(vals))
            base = float(np.mean(grouped_base[key]))
            means.append({
                "model": key[0], "family": key[1], "band": key[2], "n_widths": len(vals),
                "log_l1": err, "prior_log_l1": base, "skill": 1.0 - err / max(base, 1e-6),
            })
        return {"rows": rows, "means": means, "disagreement": pairs}


def _masks(freq: np.ndarray) -> dict[str, np.ndarray]:
    return {name: (freq >= lo) & (freq <= hi) for name, lo, hi in BANDS}


def score_split(models: dict, cache: Cache, split: str, device, freq: np.ndarray) -> dict:
    records = eval_records(
        cache, split, iid_limit=None, positions=1,
        families=("iid", "ood_dipping", "ood_three_layer"),
    )
    if split == "train":
        records = _limit_train(cache, records)
    loader = DataLoader(
        WindowSet(cache, "metric", records, STAGE_UNIFIED, seed=0, mirror=False),
        batch_size=4, shuffle=False, num_workers=0,
    )
    masks = _masks(freq)
    bucket = _Bucket()
    seen = 0
    names = list(models)
    for x, query, tf, prior, valid, info, _xa in loader:
        x = x.to(device)
        query = query.to(device)
        gs = {}
        with torch.inference_mode():
            for name in names:
                gs[name] = models[name](x, query).float()
            for label, members in ENS.items():
                gs[label] = torch.stack([gs[m] for m in members], dim=0).mean(0)
        g_np = {name: value.cpu().numpy() for name, value in gs.items()}
        valid_np = valid.numpy()
        info_np = info.numpy()
        tf_np = tf.numpy()
        prior_np = prior.numpy()
        for b in range(valid_np.shape[0]):
            stations = np.flatnonzero(valid_np[b])
            if stations.size == 0:
                continue
            family = _family(info_np[b])
            width = int(info_np[b, 2])
            true = tf_np[b, stations]
            base = prior_np[b, stations]
            pred = {name: g_np[name][b, stations] + base for name in g_np}
            for name, log_hat in pred.items():
                for band, mask in masks.items():
                    err = float(np.mean(np.abs(log_hat[:, mask] - true[:, mask])))
                    pri = float(np.mean(np.abs(base[:, mask] - true[:, mask])))
                    bucket.add(name, family, band, width, err, pri, int(stations.size))
                    if family.startswith("manifold_") and not family.startswith("manifold_blind"):
                        bucket.add(name, "manifold", band, width, err, pri, int(stations.size))
            for a, bname in PAIRS:
                delta = np.abs(g_np[a][b, stations] - g_np[bname][b, stations])
                for band, mask in masks.items():
                    bucket.add_pair(a, bname, family, band, float(delta[:, mask].mean()), int(stations.size))
            seen += int(stations.size)
        if seen and seen % 2000 < valid_np.shape[0] * 8:
            print(f"  {split} stations {seen}", flush=True)
    out = bucket.table()
    out["split"] = split
    out["n_windows"] = len(records)
    out["forward"] = "float32"
    return out


def _power(series: np.ndarray) -> tuple[float, float, float]:
    series = np.asarray(series, dtype=np.float64)
    series = series - series.mean()
    power = np.abs(np.fft.rfft(series)) ** 2
    freq = np.fft.rfftfreq(series.size, d=1.0)
    keep = freq > 0
    total = float(power[keep].sum())
    period = 1.0 / freq
    at4 = keep & (np.abs(period - 4.0) <= 0.15)
    short = keep & (period < 10.0)
    step2 = float(np.mean(np.abs(series[2:] - series[:-2])))
    step4 = float(np.mean(np.abs(series[4:] - series[:-4])))
    return float(power[at4].sum() / max(total, 1e-30)), float(power[short].sum() / max(total, 1e-30)), step2 / max(step4, 1e-12)


def sawtooth(models: dict, cache: Cache, device, freq: np.ndarray) -> dict:
    targets = [1.0, 2.5, 5.0, 8.0]
    k_of = {f"{hz:g}": int(np.argmin(np.abs(freq - hz))) for hz in targets}
    rows = []

    def _predict(vs, zeta, n_active, f1, f2, width):
        x = torch.from_numpy(encode("metric", vs, zeta, n_active, f1_hz=f1, f2_hz=f2)).unsqueeze(0).to(device)
        cols = torch.tensor(query_columns("metric", np.arange(width, dtype=np.float64), width), dtype=torch.float32).unsqueeze(0).to(device)
        out = {}
        with torch.inference_mode():
            for name, model in models.items():
                out[name] = model(x, cols).float()[0].cpu().numpy()
        for label, members in ENS.items():
            out[label] = np.mean([out[m] for m in members], axis=0)
        return out

    for path in sorted(Path("/tmp/dense_recorders").glob("time_w*.h5")):
        with h5py.File(path, "r") as handle:
            params = handle["params"].attrs
            pad = int(params["pad_m"])
            interior = int(params["lx_interior_m"])
            vs = np.asarray(handle["Vs_realization_2D"][:, pad:pad + interior], dtype=np.float64)
            zeta = np.asarray(handle["Damping_zeta"][:, pad:pad + interior], dtype=np.float64)
            rock = np.asarray(handle["rock_mask"][:, pad:pad + interior])
            f1 = float(params["rayleigh_f1_hz"])
            f2 = float(params["rayleigh_f2_hz"])
        hits = [int(np.flatnonzero(rock[:, j])[0]) if np.any(rock[:, j]) else rock.shape[0] for j in range(interior)]
        pred = _predict(vs, zeta, int(max(hits)), f1, f2, interior)
        for name, arr in pred.items():
            for label, k in k_of.items():
                p4, short, ratio = _power(arr[:, k])
                rows.append({"field": path.stem, "model": name, "hz": label, "p4": p4, "lt10": short, "d2_over_d4": ratio})
        print(f"  sawtooth {path.stem}", flush=True)

    flats = cache.runs("val", ("iid",))
    picked = []
    seen = set()
    for flat in flats:
        camp, j = cache.campaign(int(flat))
        group = int(camp.group[j])
        if group in seen:
            continue
        seen.add(group)
        picked.append(int(flat))
        if len(picked) == 3:
            break
    for flat in picked:
        camp, j = cache.campaign(flat)
        n_active = int(camp.n_active[j])
        vs = np.asarray(camp.vs[j, :n_active], dtype=np.float64)
        zeta = np.asarray(camp.zeta[j, :n_active], dtype=np.float64)
        pred = _predict(vs, zeta, n_active, float(camp.f1[j]), float(camp.f2[j]), vs.shape[1])
        for name, arr in pred.items():
            for label, k in k_of.items():
                p4, short, ratio = _power(arr[:, k])
                rows.append({"field": f"iid_val_g{int(camp.group[j])}", "model": name, "hz": label, "p4": p4, "lt10": short, "d2_over_d4": ratio})
        print(f"  sawtooth iid group {int(camp.group[j])}", flush=True)
    return {"rows": rows, "gate": "p4 below 0.01 and d2_over_d4 below 1 at 1, 2.5, 5 and 8 Hz"}


def case7_score(models: dict, device) -> dict:
    rows = []
    curves = {}
    order = list(models) + list(ENS)
    loaded = {name: models[name] for name in models}
    for crop in case7.CROPS:
        name = crop["name"]
        for width in (500, 1000, 1500, 2000):
            if width == 2000:
                mesh = case7.pr._window_from_full(float(crop["center_x_m"]), width)
                sensor = mesh.pop("sensor")
            else:
                path = case7.pr._crop_path(crop["sample_id"], width)
                if path is None:
                    continue
                mesh = case7.pr._read_mesh(path)
                sensor = mesh["vs"].shape[1] // 2
            preds = {}
            freq = truth = tf1d = None
            for label, model in loaded.items():
                freq, truth, tf1d, pred, _i = case7._forward(model, device, mesh, sensor)
                preds[label] = np.log(np.maximum(pred, 1e-6))
            log_t = np.log(np.maximum(truth, 1e-6))
            log_d = np.log(np.maximum(tf1d, 1e-6))
            for label, members in ENS.items():
                preds[label] = np.mean([preds[m] for m in members], axis=0)
            for label in order:
                for band, lo, hi in (("mid", 0.4, 2.5), ("full", 0.1, 10.0)):
                    sl = (freq >= lo) & (freq <= hi)
                    err = float(np.abs(preds[label][:, sl] - log_t[:, sl]).mean())
                    base = float(np.abs(log_d[:, sl] - log_t[:, sl]).mean())
                    gap = {}
                    if label == "ens_seed":
                        gap["seed_disagreement"] = float(np.abs(preds["P128"][:, sl] - preds["P128s1"][:, sl]).mean())
                    if label == "P128c":
                        gap["vs_p128"] = float(np.abs(preds["P128c"][:, sl] - preds["P128"][:, sl]).mean())
                    rows.append({
                        "site": name, "width": width, "model": label, "band": band,
                        "log_l1": err, "skill": 1.0 - err / max(base, 1e-6), **gap,
                    })
            print(f"  case7 {name} {width}", flush=True)
            curves[(name, width)] = True
    return {"rows": rows, "n": len(curves), "forward": "float32"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}", flush=True)
    models = {}
    stamps = {}
    for name in MODELS:
        models[name], stamps[name] = load_fp32(ROOT / name, device)
        print(f"loaded {name} epoch {stamps[name]['epoch']} val {stamps[name]['val_score']}", flush=True)
    cache = Cache(REPO / "data" / "vardomain" / "cache")
    freq = np.asarray(cache.freq, dtype=np.float64)
    header = {"checkpoints": stamps, "forward": "float32"}

    print("validation", flush=True)
    val = score_split(models, cache, "val", device, freq)
    val.update(header)
    (OUT / "audit_ensemble_val.json").write_text(json.dumps(val))
    print("wrote audit_ensemble_val.json", flush=True)

    print("train subsample", flush=True)
    train = score_split(models, cache, "train", device, freq)
    train.update(header)
    (OUT / "audit_train.json").write_text(json.dumps(train))
    print("wrote audit_train.json", flush=True)

    print("case 7", flush=True)
    c7 = case7_score(models, device)
    c7.update(header)
    (OUT / "case7_ensemble.json").write_text(json.dumps(c7))
    print("wrote case7_ensemble.json", flush=True)

    print("sawtooth", flush=True)
    tooth = sawtooth(models, cache, device, freq)
    tooth.update(header)
    (OUT / "sawtooth.json").write_text(json.dumps(tooth))
    print("wrote sawtooth.json", flush=True)


if __name__ == "__main__":
    main()
