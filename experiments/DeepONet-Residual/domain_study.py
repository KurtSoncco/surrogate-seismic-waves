#!/usr/bin/env python3
"""Architecture / operator / domain-mix study (P0–P4 + arch bake-off)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

import config
from data import CombinedResidualDataset, ResidualDeepONetDataset
from domain_splits import ensure_splits, load_split
from ood_signed_cache import (
    build_ood_signed_cache,
    cache_dir_for,
    materialize_n1000_from_n2000,
)
from train import (
    apply_norms,
    evaluate,
    train_from_datasets,
    _r2,
    _rel_l2,
    _pearson_across_freq,
)

RESULTS = config.RESULTS_DIR / "domain_study"
IID_CACHE = config.CACHE_DIR / "n1000_seed42"
E2_CKPT = (
    config.CHECKPOINT_DIR / "single_resunet_full_R_nom_n2000_seed42_nf200_seed42.pt"
)


def _set_local_ood_env() -> None:
    local = Path(__file__).resolve().parents[2] / "data" / "gifno_screen"
    if (local / "ood_dipping").is_dir():
        os.environ.setdefault("GIFNO_OOD_DIPPING", str(local / "ood_dipping"))
    if (local / "ood_three_layer").is_dir():
        os.environ.setdefault("GIFNO_OOD_THREE_LAYER", str(local / "ood_three_layer"))


def _tf_metrics(tf2d: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(tf2d, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    n_rec, n_freq = y.shape[-2], y.shape[-1]
    return {
        "r2": _r2(y.ravel(), p.ravel()),
        "rel_l2": _rel_l2(y.ravel(), p.ravel()),
        "pearson_freq": _pearson_across_freq(
            y.ravel(), p.ravel(), n_rec=n_rec, n_freq=n_freq
        ),
    }


def _slice_cache(cache: Path, idx: np.ndarray, key: str) -> np.ndarray:
    return np.load(cache / key, mmap_mode="r")[idx]


def operator_bakeoff(splits: dict[str, Path]) -> dict[str, Any]:
    """Haskell priors + frozen IID R-hat on each held-out test slice."""
    report: dict[str, Any] = {}
    iid = load_split(splits["iid"])
    iid_idx = iid["test"]
    tf_all = np.load(config.TF_PER_SAMPLE_PATH, mmap_mode="r")
    sidx = np.load(IID_CACHE / "sample_indices.npy")[iid_idx]
    tf2d = np.asarray(tf_all[sidx])
    nom = _slice_cache(IID_CACHE, iid_idx, "tf1d_nom.npy")
    col = _slice_cache(IID_CACHE, iid_idx, "tf1d_col.npy")
    report["iid"] = {
        "n": int(len(iid_idx)),
        "haskell_nom": _tf_metrics(tf2d, nom),
        "haskell_col": _tf_metrics(tf2d, col),
    }

    for name in ("ood_dipping", "ood_three_layer"):
        cache = cache_dir_for(name)
        sp = load_split(splits[name])
        te = sp["test"]
        tf2d = _slice_cache(cache, te, "tf2d.npy")
        pack = {
            "n": int(len(te)),
            "haskell_nom": _tf_metrics(tf2d, _slice_cache(cache, te, "tf1d_nom.npy")),
            "haskell_col": _tf_metrics(tf2d, _slice_cache(cache, te, "tf1d_col.npy")),
        }
        nom1_path = cache / "tf1d_nom1.npy"
        if nom1_path.is_file() and name == "ood_three_layer":
            pack["haskell_nom1_misspecified"] = _tf_metrics(
                tf2d, _slice_cache(cache, te, "tf1d_nom1.npy")
            )
        report[name] = pack

    if E2_CKPT.is_file():
        report["frozen_e2_rhat"] = _eval_frozen_rhat(splits)
    return report


def _eval_frozen_rhat(splits: dict[str, Path]) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from eval_ood import _load_residual_model
    from torch.utils.data import DataLoader

    model, blob, stats, trunk_set = _load_residual_model(E2_CKPT, device)
    out: dict[str, Any] = {}
    specs = [
        ("iid", IID_CACHE, load_split(splits["iid"])["test"]),
        (
            "ood_dipping",
            cache_dir_for("ood_dipping"),
            load_split(splits["ood_dipping"])["test"],
        ),
        (
            "ood_three_layer",
            cache_dir_for("ood_three_layer"),
            load_split(splits["ood_three_layer"])["test"],
        ),
    ]
    for name, cache, idx in specs:
        ds = ResidualDeepONetDataset(
            cache, idx, target="R_nom", trunk_set="full", n_freq=config.N_FREQ_EVAL
        )
        apply_norms(ds, stats)
        loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False)
        out[name] = evaluate(
            model,
            loader,
            device,
            blob.get("branch_mode", "single"),
            stats,
            n_rec=ds.n_rec,
            n_freq=len(ds.f_idx),
        )
        print(
            f"[frozen E2] {name} r2_R={out[name]['r2_R']:.3f} "
            f"rel_l2_TF={out[name]['rel_l2_TF']:.3f}",
            flush=True,
        )
    return out


def _ds(
    cache: Path,
    idx: np.ndarray,
    *,
    n_freq: int,
    serial: bool = False,
) -> ResidualDeepONetDataset:
    return ResidualDeepONetDataset(
        cache,
        idx,
        target="R_nom",
        trunk_set="full",
        n_freq=n_freq,
        serial_tf1d=serial,
    )


def _run_protocol(
    name: str,
    train_parts: list[tuple[str, Path, np.ndarray]],
    val_parts: list[tuple[str, Path, np.ndarray]],
    test_parts: dict[str, tuple[Path, np.ndarray]],
    *,
    n_freq_train: int,
    n_freq_eval: int,
    encoder: str,
    branch_mode: str,
    serial: bool,
    init_ckpt: Path | None,
    lr: float,
    patience: int,
    epochs: int,
) -> dict[str, Any]:
    train_ds = CombinedResidualDataset(
        [_ds(c, i, n_freq=n_freq_train, serial=serial) for _, c, i in train_parts],
        domain_names=[name for name, _, _ in train_parts],
    )
    val_ds = CombinedResidualDataset(
        [_ds(c, i, n_freq=n_freq_train, serial=serial) for _, c, i in val_parts],
        domain_names=[name for name, _, _ in val_parts],
    )
    extra = {
        dname: _ds(c, i, n_freq=n_freq_eval, serial=serial)
        for dname, (c, i) in test_parts.items()
    }
    return train_from_datasets(
        train_ds=train_ds,
        val_ds=val_ds,
        extra_tests=extra,
        target="R_nom",
        branch_mode=branch_mode,  # type: ignore[arg-type]
        trunk_set="full",
        epochs=epochs,
        batch_size=config.BATCH_SIZE,
        lr=lr,
        seed=config.SEED,
        run_name=name,
        patience=patience,
        field_encoder=encoder,  # type: ignore[arg-type]
        n_freq_train=n_freq_train,
        n_freq_eval=n_freq_eval,
        init_ckpt=init_ckpt,
        serial_tf1d=serial,
    )


def run_protocols(splits: dict[str, Path]) -> dict[str, Any]:
    iid = load_split(splits["iid"])
    dip = load_split(splits["ood_dipping"])
    tl = load_split(splits["ood_three_layer"])
    iid_c, dip_c, tl_c = (
        IID_CACHE,
        cache_dir_for("ood_dipping"),
        cache_dir_for("ood_three_layer"),
    )
    tests = {
        "iid": (iid_c, iid["test"]),
        "ood_dipping": (dip_c, dip["test"]),
        "ood_three_layer": (tl_c, tl["test"]),
    }
    kw = dict(
        n_freq_train=200,
        n_freq_eval=1000,
        encoder="resunet",
        branch_mode="single",
        serial=False,
        lr=config.LR,
        patience=config.PATIENCE,
        epochs=config.EPOCHS,
        test_parts=tests,
    )
    out: dict[str, Any] = {}
    out["P0_iid"] = _run_protocol(
        "P0_iid_resunet",
        [("iid", iid_c, iid["train"])],
        [("iid", iid_c, iid["val"])],
        init_ckpt=None,
        **kw,
    )
    out["P1_two_layer"] = _run_protocol(
        "P1_iid_dipping_resunet",
        [("iid", iid_c, iid["train"]), ("dip", dip_c, dip["train"])],
        [("iid", iid_c, iid["val"]), ("dip", dip_c, dip["val"])],
        init_ckpt=None,
        **kw,
    )
    p1_ckpt = Path(out["P1_two_layer"]["checkpoint"])
    out["P2_finetune_3l"] = _run_protocol(
        "P2_finetune_three_layer_resunet",
        [("tl", tl_c, tl["train"])],
        [("tl", tl_c, tl["val"])],
        init_ckpt=p1_ckpt,
        **{**kw, "lr": 1e-4, "patience": 30},
    )
    out["P3_mix"] = _run_protocol(
        "P3_mix_resunet",
        [
            ("iid", iid_c, iid["train"]),
            ("dip", dip_c, dip["train"]),
            ("tl", tl_c, tl["train"]),
        ],
        [
            ("iid", iid_c, iid["val"]),
            ("dip", dip_c, dip["val"]),
            ("tl", tl_c, tl["val"]),
        ],
        init_ckpt=None,
        **kw,
    )
    out["P4_three_only"] = _run_protocol(
        "P4_three_layer_only_resunet",
        [("tl", tl_c, tl["train"])],
        [("tl", tl_c, tl["val"])],
        init_ckpt=None,
        **kw,
    )
    return out


def _pick_winner(protocols: dict[str, Any]) -> str:
    """Best 3-layer test rel_l2_TF without collapsing IID (r2_R > 0)."""
    best, best_key = 1e9, "P3_mix"
    for key, res in protocols.items():
        by = res.get("test_by_domain", {})
        tl = by.get("ood_three_layer", {})
        iid = by.get("iid", {})
        if not tl or not iid:
            continue
        if iid.get("r2_R", -1) < -0.05:
            continue
        score = float(tl.get("rel_l2_TF", 9))
        if score < best:
            best, best_key = score, key
    return best_key


def run_arch(splits: dict[str, Path], protocol_key: str) -> dict[str, Any]:
    iid = load_split(splits["iid"])
    dip = load_split(splits["ood_dipping"])
    tl = load_split(splits["ood_three_layer"])
    iid_c, dip_c, tl_c = (
        IID_CACHE,
        cache_dir_for("ood_dipping"),
        cache_dir_for("ood_three_layer"),
    )
    tests = {
        "iid": (iid_c, iid["test"]),
        "ood_dipping": (dip_c, dip["test"]),
        "ood_three_layer": (tl_c, tl["test"]),
    }
    mix_train = [
        ("iid", iid_c, iid["train"]),
        ("dip", dip_c, dip["train"]),
        ("tl", tl_c, tl["train"]),
    ]
    mix_val = [
        ("iid", iid_c, iid["val"]),
        ("dip", dip_c, dip["val"]),
        ("tl", tl_c, tl["val"]),
    ]
    if protocol_key == "P1_two_layer":
        train, val = mix_train[:2], mix_val[:2]
    elif protocol_key == "P0_iid":
        train, val = [("iid", iid_c, iid["train"])], [("iid", iid_c, iid["val"])]
    elif protocol_key == "P4_three_only":
        train, val = [("tl", tl_c, tl["train"])], [("tl", tl_c, tl["val"])]
    elif protocol_key == "P2_finetune_3l":
        train, val = mix_train[:2], mix_val[:2]
    else:
        train, val = mix_train, mix_val
        protocol_key = "P3_mix"

    kw = dict(
        n_freq_train=200,
        n_freq_eval=1000,
        lr=config.LR,
        patience=config.PATIENCE,
        epochs=config.EPOCHS,
        test_parts=tests,
        init_ckpt=None,
    )
    specs = [
        ("conv", "conv", "single", False),
        ("multi", "resunet", "multi", False),
        ("serial", "resunet", "single", True),
    ]
    out: dict[str, Any] = {}
    for tag, encoder, branch, serial in specs:
        if protocol_key == "P2_finetune_3l":
            stage1 = _run_protocol(
                f"arch_{tag}_P1_stage",
                mix_train[:2],
                mix_val[:2],
                encoder=encoder,
                branch_mode=branch,
                serial=serial,
                **kw,
            )
            out[tag] = _run_protocol(
                f"arch_{tag}_{protocol_key}",
                [("tl", tl_c, tl["train"])],
                [("tl", tl_c, tl["val"])],
                encoder=encoder,
                branch_mode=branch,
                serial=serial,
                **{
                    **kw,
                    "lr": 1e-4,
                    "patience": 30,
                    "init_ckpt": Path(stage1["checkpoint"]),
                },
            )
            out[tag]["stage1"] = {
                "checkpoint": stage1.get("checkpoint"),
                "test_by_domain": stage1.get("test_by_domain"),
            }
        else:
            out[tag] = _run_protocol(
                f"arch_{tag}_{protocol_key}",
                train,
                val,
                encoder=encoder,
                branch_mode=branch,
                serial=serial,
                **kw,
            )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-cache", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-operator", action="store_true")
    p.add_argument("--skip-protocols", action="store_true")
    p.add_argument("--winner", type=str, default=None)
    p.add_argument("--force-cache", action="store_true")
    args = p.parse_args()
    _set_local_ood_env()
    RESULTS.mkdir(parents=True, exist_ok=True)
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("[study] writing splits", flush=True)
    split_paths = ensure_splits()
    if not args.skip_cache:
        materialize_n1000_from_n2000()
        for name in ("ood_dipping", "ood_three_layer"):
            build_ood_signed_cache(name, force=args.force_cache)

    ops_path = RESULTS / "operator_bakeoff.json"
    if args.skip_operator and ops_path.is_file():
        ops = json.loads(ops_path.read_text())
        print("[study] reuse operator_bakeoff.json", flush=True)
    else:
        print("[study] operator bake-off", flush=True)
        ops = operator_bakeoff(split_paths)
        ops_path.write_text(json.dumps(ops, indent=2, default=str))
        print(json.dumps(ops, indent=2, default=str)[:2000], flush=True)

    if args.skip_train:
        return

    proto_path = RESULTS / "protocols.json"
    if args.skip_protocols and proto_path.is_file():
        protocols = json.loads(proto_path.read_text())
        print("[study] reuse protocols.json", flush=True)
    else:
        print("[study] domain protocols P0–P4", flush=True)
        protocols = run_protocols(split_paths)
        proto_path.write_text(json.dumps(protocols, indent=2, default=str))
    winner = args.winner or _pick_winner(protocols)
    print(f"[study] winner protocol={winner}", flush=True)
    print("[study] architecture bake-off", flush=True)
    arch = run_arch(split_paths, winner)
    (RESULTS / "architectures.json").write_text(json.dumps(arch, indent=2, default=str))
    summary = {"operator": ops, "winner_protocol": winner, "protocols": {}, "arch": {}}
    for k, v in protocols.items():
        summary["protocols"][k] = {
            "checkpoint": v.get("checkpoint"),
            "test_by_domain": v.get("test_by_domain"),
            "epochs_ran": v.get("epochs_ran"),
        }
    for k, v in arch.items():
        summary["arch"][k] = {
            "checkpoint": v.get("checkpoint"),
            "test_by_domain": v.get("test_by_domain"),
            "epochs_ran": v.get("epochs_ran"),
        }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[study] done → {RESULTS / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
