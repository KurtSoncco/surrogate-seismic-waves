#!/usr/bin/env python3
"""Smoke-train direct |TF| path without GIFNO corpus (synthetic recorders).

Verifies SmoothL1 (+peak+spec) + Fourier FNO/DeepONet train loop end-to-end.
Not a scientific result — only a pipeline gate before Lambda.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

_EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXP))
import config

config.setup_import_paths()

from losses_th import THFNOLoss  # noqa: E402
from model import GatedDeltaModel  # noqa: E402


def main():
    device = config.DEVICE
    B, nz, nx, nf = 4, 32, 64, 48
    rec = np.linspace(0, nx - 1, 7, dtype=int)
    freq = np.logspace(-1, 1, nf)
    model = GatedDeltaModel(
        in_channels=6,
        latent_channels=8,
        n_freq=nf,
        nx=nx,
        fno_modes=(4, 4),
        num_fno_layers=1,
        deeponet_dim=8,
        use_fourier=True,
        n_fourier=4,
        physics_dim=3,
        predict_mode="direct",
        recorder_x=rec,
        freq=freq,
    ).to(device)

    rng = np.random.RandomState(0)
    x = torch.from_numpy(rng.randn(B, 6, nz, nx).astype(np.float32))
    haskell = torch.ones(B, nx, nf)
    target = torch.from_numpy(np.abs(rng.randn(B, nx, nf)).astype(np.float32) + 0.2)
    mask = torch.zeros(B, nx)
    mask[:, rec] = 1.0
    cov = torch.full((B,), 0.2)
    dip = torch.full((B,), 0.05)
    physics = torch.tensor([[0.2, 20.0, 40.0]] * B, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(x, haskell, target, mask, cov, dip, physics), batch_size=2
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = THFNOLoss(freq=torch.as_tensor(freq, dtype=torch.float32)).to(device)

    losses = []
    for _epoch in range(3):
        for batch in loader:
            xb, hb, tb, mb, cv, dp, ph = [t.to(device) for t in batch]
            pred = model(xb, hb, cv, dp, physics=ph)
            loss, _parts = loss_fn(pred, tb, mb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            assert torch.isfinite(pred).all()
            assert (pred >= 0).all()

    out = config.RESULTS_SAVE_DIR / "smoke_direct_train.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "predict_mode": "direct",
        "final_loss": losses[-1],
        "n_steps": len(losses),
        "ok": bool(np.isfinite(losses[-1])),
    }
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
