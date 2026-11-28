"""AI (figstyle.STAGE) and 1D |TF| at the 233 SVLAB stations of Treasure Island.

Same windows, damping, and prior as GIFNO-VARDOMAIN/score_svlab_stage1.py;
writes the curves so slide 15 can score any band. Checks the 0.1-2.5 Hz
Pearson against the stored svlab_metrics.json.

    python prep_svlab.py
"""

from __future__ import annotations

import json
import sys

import h5py
import numpy as np
import torch

import figstyle as fs

sys.path.insert(0, str(fs.REPO / "experiments" / "GIFNO-VARDOMAIN"))
import score_svlab_stage1 as s1  # noqa: E402
from provenance import full_stamp, write_sidecar  # noqa: E402

OUT = fs.DATA / f"svlab_{fs.STAGE}_tf.npz"
STORED = fs.RESULTS / "svlab_metrics.json"


def main() -> None:
    pack = s1.ensure_cache(force=False)
    x = s1.window_center_x()
    nodes = np.array([s1.window_node_index(float(v)) for v in x], dtype=np.int64)
    freq_gt, tf_gt, _scale = s1.outcrop_reference_transfer_function(pack, nodes)
    freq_m = np.load(s1.CACHE_FREQ)
    with h5py.File(s1.FULL_QVS, "r") as handle:
        vs = np.asarray(handle["Vs_realization_2D"][:], dtype=np.float64)
        zeta = s1.applied_zeta(handle)
        f1 = float(handle["params"].attrs["damping_freq_first"])
        f2 = float(handle["params"].attrs["damping_freq_second"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = s1.load_model(fs.CHECKPOINT, device)
    model.eval()
    stamp = full_stamp(fs.CHECKPOINT, device, bool(cfg.get("amp", True)))
    query = torch.tensor(s1.query_columns("metric", np.array([s1.WIDTH / 2.0]), s1.WIDTH),
                         dtype=torch.float32).view(1, 1)
    pred = np.empty((x.size, freq_gt.size))
    prior = np.empty_like(pred)
    with torch.no_grad():
        for i, center in enumerate(x):
            c0, n, _shift = s1.span_columns(float(center), s1.WIDTH)
            vs_w, zeta_w = vs[:, c0 : c0 + n], zeta[:, c0 : c0 + n]
            tf1d = s1._column_tf(freq_m, vs_w[:, n // 2], zeta_w[:, n // 2], f1, f2)
            enc = torch.from_numpy(s1.encode("metric", vs_w, zeta_w, vs_w.shape[0], f1_hz=f1, f2_hz=f2)).unsqueeze(0)
            g = model(enc.to(device), query.to(device)).float().cpu().numpy()[0, 0]
            prior[i] = np.interp(freq_gt, freq_m, tf1d)
            pred[i] = np.interp(freq_gt, freq_m, tf1d * np.exp(g))
    np.savez_compressed(OUT, x=x, freq=freq_gt, tf_gt=tf_gt, pred=pred, prior=prior)
    write_sidecar(OUT, {**stamp, "n_stations": int(x.size), "mesh": str(s1.FULL_QVS)})

    stored = json.loads(STORED.read_text())["pearson_vs_x"]
    for name, curve in ((fs.STAGE_LABEL, pred), ("Column", prior)):
        again = s1._along(freq_gt, curve, tf_gt, 0.1, 2.5)
        diff = np.max(np.abs(again - np.asarray(stored[name]["0.1–2.5 Hz"])))
        print(f"{name}: max |Pearson 0.1-2.5 Hz - stored| = {diff:.2e}")
    print(f"wrote {OUT} ({x.size} stations, device {device})")


if __name__ == "__main__":
    main()
