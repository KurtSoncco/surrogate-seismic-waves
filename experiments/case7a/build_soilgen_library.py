#!/usr/bin/env python3
"""Sample constrained-flow 1D columns from gen-ai-soil-profiles.

Runs in a subprocess with PYTHONPATH set to the sibling repo so its
``experiments`` package does not collide with this one. Writes
``data/case7a/soilgen_library.npz``.

    python build_soilgen_library.py --n 64
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from soilgen_columns import DEFAULT_LIBRARY, SOILGEN_CKPT, SOILGEN_ROOT


def _sampler_source() -> str:
    return r'''
import sys, types
sys.modules.setdefault("torchinfo", types.SimpleNamespace(summary=lambda *a, **k: None))
import json
from pathlib import Path

import numpy as np
import torch

from experiments.flow_matching_simplified.data import FlowMatchingDataLoader
from experiments.flow_matching_simplified.eval_corrected import vs_from_travel_time
from experiments.flow_matching_simplified.train_constrained import (
    ConstrainedTokens,
    build_model,
    fit_toro,
    load_split,
    sample,
)
from experiments.flow_matching_simplified.site_metrics import vs_z_or_h

ckpt = Path(__import__("os").environ["SOILGEN_CKPT"])
n = int(__import__("os").environ["SOILGEN_N"])
seed = int(__import__("os").environ.get("SOILGEN_SEED", "42"))
out = Path(__import__("os").environ["SOILGEN_OUT"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_seq, _, _ = load_split()
data = ConstrainedTokens(train_seq)
toro = fit_toro(train_seq)
model = build_model()
state = torch.load(ckpt, map_location=device, weights_only=False)
model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
model.to(device)
seqs = sample(model, data, n=n, seed=seed, device=device, steps=50, toro_fit=toro)
max_n = 20
h = np.zeros(n)
vs_z = np.zeros(n)
n_lay = np.zeros(n, dtype=np.int32)
bounds = np.zeros((n, max_n + 1))
vs = np.zeros((n, max_n))
kept = 0
for seq in seqs:
    z0, z1, v = vs_from_travel_time(seq)
    k = int(v.size)
    if k < 2:
        continue
    b = np.concatenate([z0[:1], z1])
    h[kept] = float(b[-1])
    vs_z[kept] = float(vs_z_or_h(b, v))
    n_lay[kept] = k
    bounds[kept, : k + 1] = b
    vs[kept, :k] = v
    kept += 1
np.savez_compressed(out, h=h[:kept], vs_z=vs_z[:kept], n=n_lay[:kept], bounds=bounds[:kept], vs=vs[:kept])
print(f"wrote {kept} columns -> {out}")
'''


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=DEFAULT_LIBRARY)
    ap.add_argument("--ckpt", type=Path, default=SOILGEN_CKPT)
    args = ap.parse_args()
    if not SOILGEN_ROOT.is_dir():
        raise FileNotFoundError(f"gen-ai-soil-profiles not at {SOILGEN_ROOT}")
    if not args.ckpt.is_file():
        raise FileNotFoundError(args.ckpt)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOILGEN_ROOT)
    env["SOILGEN_CKPT"] = str(args.ckpt)
    env["SOILGEN_N"] = str(args.n)
    env["SOILGEN_SEED"] = str(args.seed)
    env["SOILGEN_OUT"] = str(args.out)
    py = sys.executable
    subprocess.check_call([py, "-c", _sampler_source()], cwd=str(SOILGEN_ROOT), env=env)


if __name__ == "__main__":
    main()
