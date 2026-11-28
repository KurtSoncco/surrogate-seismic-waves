#!/usr/bin/env python3
"""Column and state of practice on the Lysmer base OpenSees actually runs.

The mesh has a finite bedrock and an ASDAbsorbingBoundary2D under it. The
Ricker is the incident wave at that dashpot. |TF| is surface / incident, which
is twice the stored outcrop ratio surface / (2 A_up). OpenSees labels in the
cache were divided by the campaign scale c so they would match the outcrop
ratio; multiplying them back by c restores surface / injected wave.

A real dashpot (elastic rho*Vs) and the complex half-space used by the stored
column differ by about 6e-4 in log on these rock dampings, so the stored
column and the matched practice curves are scaled by 2 rather than recomputed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

from bootstrap_ci import bootstrap_cluster_statistic  # noqa: E402
from center_curves import FAMILIES, test_frame  # noqa: E402
from metrics import anderson, pearson  # noqa: E402

OUT = HERE / "03_state_of_practice"
# relabel_outcrop.py scale, fixed 1310.72 s window. surface/injected = c * stored.
SCALE = {"iid": 1.982620120048523, "ood_dipping": 1.9776999950408936}
METHODS = ("Column", "Pretell", "Passeri", "Dmult", "Toro")


def _score(freq, truth, curve, f0) -> dict[str, np.ndarray]:
    band = (freq >= 0.4) & (freq <= 2.5)
    whole = (freq >= 0.1) & (freq <= 10.0)
    log_c = np.log(np.clip(curve, 1e-12, None))
    log_t = np.log(np.clip(truth, 1e-12, None))
    return {
        "pearson": np.array([pearson(curve[i], truth[i]) for i in range(truth.shape[0])]),
        "anderson": np.array([anderson(freq, truth[i], curve[i], f0[i]) for i in range(truth.shape[0])]),
        "log_l1_mid": np.mean(np.abs(log_c[:, band] - log_t[:, band]), axis=1),
        "log_l1_full": np.mean(np.abs(log_c[:, whole] - log_t[:, whole]), axis=1),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, family in FAMILIES.items():
        frame = test_frame(family)
        matched = np.load(OUT / f"matched_{family}.npz")
        order = {int(run): i for i, run in enumerate(np.asarray(matched["run"]))}
        c = SCALE[family]
        truth = frame["opensees"] * c
        curves = {"Column": frame["column"] * 2.0}
        for method in METHODS[1:]:
            curves[method] = np.stack([
                np.asarray(matched[method.lower()][order[int(run)]], dtype=np.float64) for run in frame["run"]
            ]) * 2.0
        print(f"{name}  c={c:.4f}  truth/column median {np.median(truth / curves['Column']):.4f}", flush=True)
        for method in METHODS:
            scored = _score(frame["freq"], truth, curves[method], frame["f0"])
            stat_p = bootstrap_cluster_statistic(scored["pearson"], frame["group"], np.median, rng=np.random.default_rng(0))
            stat_a = bootstrap_cluster_statistic(scored["anderson"], frame["group"], np.median, rng=np.random.default_rng(0))
            site = pd.DataFrame({"value": scored["pearson"], "group": frame["group"]}).groupby("group")["value"].median()
            rows.append({
                "domain": name,
                "method": method,
                "n_records": int(frame["run"].size),
                "n_sites": stat_p["n_clusters"],
                "pearson_record": stat_p["point"],
                "pearson_low": stat_p["low"],
                "pearson_high": stat_p["high"],
                "pearson_site": float(site.median()),
                "anderson_record": stat_a["point"],
                "anderson_low": stat_a["low"],
                "anderson_high": stat_a["high"],
                "log_l1_0p4_2p5": float(np.median(scored["log_l1_mid"])),
                "log_l1_0p1_10": float(np.median(scored["log_l1_full"])),
                "input_scale_c": c,
            })
    table = pd.DataFrame(rows)
    dest = OUT / "table_asda_elastic.csv"
    table.to_csv(dest, index=False)
    print(table.to_string(index=False))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
