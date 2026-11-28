#!/usr/bin/env python3
"""Case 7 width test for stage 2 and P128n128, float32.

The published figure scores 0.4–2.5 Hz because the records are 15 s. This also
scores the whole 0.1–10 Hz grid. Figures are written only for P128n128.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
VARDOMAIN = HERE.parent
sys.path.insert(0, str(VARDOMAIN))

import plot_results as pr
from case7_crops import CROPS
from evaluate import load_model
from model import load_vardomain_state, model_from_config

OUT = HERE / "p128n128"
STAGE2 = VARDOMAIN.parents[1] / "checkpoints" / "vardomain" / "early_width" / "stage2_rise50"
P128N = VARDOMAIN.parents[1] / "checkpoints" / "vardomain" / "v2_ablations" / "P128n128"
PUBLISHED = VARDOMAIN / "figures" / "stage2" / "case7_width.json"


def load_fp32(model_dir: Path, device: torch.device):
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
    model.to(device).eval()
    return model


def _metrics(pred, truth, tf1d, freq, lo: float, hi: float) -> dict:
    band = (freq >= lo) & (freq <= hi)
    log_p = np.log(np.maximum(pred[:, band], 1e-6))
    log_t = np.log(np.maximum(truth[:, band], 1e-6))
    log_d = np.log(np.maximum(tf1d[:, band], 1e-6))
    log_m = float(np.abs(log_p - log_t).mean())
    log_0 = float(np.abs(log_d - log_t).mean())
    return {
        "pearson": pr._pearson(log_p.ravel(), log_t.ravel()),
        "pearson_1d": pr._pearson(log_d.ravel(), log_t.ravel()),
        "log_l1": log_m,
        "skill": float(1.0 - log_m / max(log_0, 1e-6)),
        "n_stations": int(pred.shape[0]),
    }


def _forward(model, device, mesh: dict, sensor: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Return frequency, OpenSees |TF|, 1D |TF|, model |TF|, and the sensor row."""
    from encoding import encode, query_columns
    from labels import seiskit_input_acceleration, tf_over_input
    from section_1d import RHO, rayleigh_xi_target, surface_over_outcrop

    vs, zeta = mesh["vs"], mesh["zeta"]
    width = vs.shape[1]
    cols = np.arange(0, width, pr.STRIDE_M)
    if sensor not in cols:
        cols = np.sort(np.append(cols, sensor))
    cols = cols[(cols >= 0) & (cols < width)]
    time, dt = mesh["time"], float(mesh["time"][1] - mesh["time"][0])
    freq, raw = tf_over_input(
        mesh["acc"][:, cols],
        seiskit_input_acceleration(mesh["motion"], 0.5, time, mesh["ricker"]),
        dt,
        __import__("dense_labels")._ttf(),
        window_s=pr.WINDOW_S,
    )
    truth = raw / pr.SCALE
    tf1d = np.empty_like(truth)
    for j, col in enumerate(cols):
        xi = np.stack([rayleigh_xi_target(freq, z, mesh["f1"], mesh["f2"]) for z in zeta[:, col]])
        tf1d[j] = surface_over_outcrop(freq, vs[:, col], dz=1.0, xi=xi, rho=RHO)
    enc = torch.from_numpy(encode("metric", vs, zeta, vs.shape[0], f1_hz=mesh["f1"], f2_hz=mesh["f2"])).unsqueeze(0)
    query = torch.tensor(query_columns("metric", cols.astype(np.float64), width), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        raw_g = model(enc.to(device), query.to(device)).float().cpu().numpy()[0]
    pred = tf1d * np.exp(raw_g)
    sensor_row = int(np.where(cols == sensor)[0][0])
    return freq, truth, tf1d, pred, sensor_row


def score(model, device) -> dict:
    out = {}
    for crop in CROPS:
        name = crop["name"]
        out[name] = {}
        for width in (500, 1000, 1500, 2000):
            if width == 2000:
                mesh = pr._window_from_full(float(crop["center_x_m"]), width)
                sensor = mesh.pop("sensor")
            else:
                path = pr._crop_path(crop["sample_id"], width)
                if path is None:
                    continue
                mesh = pr._read_mesh(path)
                sensor = mesh["vs"].shape[1] // 2
            freq, truth, tf1d, pred, i_s = _forward(model, device, mesh, sensor)
            mid = _metrics(pred, truth, tf1d, freq, 0.4, 2.5)
            full = _metrics(pred, truth, tf1d, freq, 0.1, 10.0)
            curves = {"freq": freq, "truth": truth[i_s], "prior": tf1d[i_s], "pred": pred[i_s]}
            out[name][width] = {"mid": mid, "full": full, "curves": curves}
            print(
                f"{name:12} w{width} mid skill {mid['skill']:+.3f} pearson {mid['pearson']:.3f} "
                f"full skill {full['skill']:+.3f} logL1 {full['log_l1']:.3f}",
                flush=True,
            )
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}", flush=True)
    published = json.loads(PUBLISHED.read_text())
    results = {}
    for label, path in (("stage2", STAGE2), ("P128n128", P128N)):
        print(label, flush=True)
        model = load_fp32(path, device)
        # load_model is the paper path; this load skips its bfloat16 bind.
        _ = load_model
        results[label] = score(model, device)
        del model
        torch.cuda.empty_cache()
    pr.FIG_DIR = OUT
    pr.fig_case7({name: {w: {"skill": row["mid"]["skill"], "pearson": row["mid"]["pearson"], "pearson_1d": row["mid"]["pearson_1d"], "curves": row["curves"]} for w, row in widths.items()} for name, widths in results["P128n128"].items()})
    rows = []
    for name in results["P128n128"]:
        for width in results["P128n128"][name]:
            new = results["P128n128"][name][width]
            old = results["stage2"][name][width]
            pub = published[name][str(width)]
            rows.append({
                "site": name,
                "width": width,
                "stage2_skill": old["mid"]["skill"],
                "published_skill": pub["skill"],
                "p128n128_skill": new["mid"]["skill"],
                "stage2_log_l1": old["mid"]["log_l1"],
                "p128n128_log_l1": new["mid"]["log_l1"],
                "stage2_pearson": old["mid"]["pearson"],
                "p128n128_pearson": new["mid"]["pearson"],
                "stage2_full_log_l1": old["full"]["log_l1"],
                "p128n128_full_log_l1": new["full"]["log_l1"],
                "stage2_full_skill": old["full"]["skill"],
                "p128n128_full_skill": new["full"]["skill"],
            })
    payload = {
        "band_hz": "0.4-2.5 is the paper band; full is 0.1-10",
        "forward": "float32",
        "rows": rows,
    }
    (OUT / "case7_width.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
