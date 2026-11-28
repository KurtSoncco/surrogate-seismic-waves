#!/usr/bin/env python3
"""Score one finished checkpoint with the validation call used in training.

Prints the selection score and the readouts the contact arm is judged on.
``m1_rel_lin`` is the relative error of linear |TF| at the true fundamental
peak. ``m1_amp_log`` is the older log-amplitude gap and is printed beside it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from data import STAGE_UNIFIED, Cache
from evaluate import evaluate
from model import load_vardomain_state, model_from_config


def main() -> None:
    model_dir = Path(sys.argv[1])
    cache_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/vardomain/cache")
    device = torch.device(sys.argv[3] if len(sys.argv) > 3 else "cpu")
    ckpt = torch.load(model_dir / "best_model.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    modes = np.load(model_dir / "pod_modes.npy")
    mean = np.load(model_dir / "pod_mean.npy")
    pca_mean = np.load(model_dir / "pca_mean.npy")
    pca_components = np.load(model_dir / "pca_components.npy")
    model = model_from_config(cfg, modes, mean, pca_mean, pca_components)
    load_vardomain_state(model, ckpt["state_dict"])
    model.to(device).eval()
    cache = Cache(cache_dir)
    res = evaluate(
        model, cache, cfg["encoding"], "val", device,
        batch=8 if device.type == "cuda" else 4,
        workers=0,
        iid_limit=None, positions=1, amp=False,
        target=cfg["target"], families=("iid", "ood_dipping", "ood_three_layer"),
        stage=int(cfg.get("stage", STAGE_UNIFIED)),
    )

    def fam(name: str, key: str) -> float | None:
        block = res.get(f"{name}_mean") or {}
        value = block.get(key)
        return None if value is None else float(value)

    iid = fam("iid", "log_l1")
    man = fam("manifold", "log_l1")
    bands = {}
    for key, block in sorted(res.get("bins", {}).items()):
        bands[key] = {
            "n": block.get("n_stations"),
            "hz_0p1_10": block.get("log_l1"),
            "skill_0p1_10": block.get("skill_full"),
            "mid": block.get("log_l1_mid"),
            "skill_mid": block.get("skill_mid"),
            "high": block.get("log_l1_high"),
            "m1_rel_lin": block.get("m1_rel_lin"),
            "m1_amp_log": block.get("m1_amp_log"),
        }
    out = {
        "dir": model_dir.name,
        "epoch": ckpt.get("epoch"),
        "device": str(device),
        "selection_0p1_10": None if iid is None or man is None else 0.5 * (iid + man),
        "iid_0p1_10": iid,
        "iid_skill_0p1_10": fam("iid", "skill_full"),
        "manifold_0p1_10": man,
        "manifold_skill_0p1_10": fam("manifold", "skill_full"),
        "iid_mid": fam("iid", "log_l1_mid"),
        "iid_high": fam("iid", "log_l1_high"),
        "manifold_mid": fam("manifold", "log_l1_mid"),
        "manifold_high": fam("manifold", "log_l1_high"),
        "manifold/m1_rel_lin": fam("manifold", "m1_rel_lin"),
        "manifold/m1_amp_log": fam("manifold", "m1_amp_log"),
        "dip_0p1_10": fam("ood_dipping", "log_l1"),
        "dip_mid": fam("ood_dipping", "log_l1_mid"),
        "three_0p1_10": fam("ood_three_layer", "log_l1"),
        "three_mid": fam("ood_three_layer", "log_l1_mid"),
        "bins": bands,
    }
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
