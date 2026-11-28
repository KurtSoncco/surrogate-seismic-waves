#!/usr/bin/env python3
"""Write the external-audit status page. It does not rescore P128n128 or open a site."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DECISION = HERE.parents[1] / "analysis" / "preregistration" / "decision.json"
FLOORS = HERE.parents[1] / "analysis" / "preregistration" / "local_floors.json"
OUT = HERE / "08_external_audit" / "README.md"


def main() -> None:
    decision = json.loads(DECISION.read_text())
    floors = json.loads(FLOORS.read_text())
    if decision["external_labels_opened"] or decision["confirmatory_training_opened"]:
        raise SystemExit("status page refuses an opened audit that this writer did not run")
    bands = floors["output_pod_floor_by_band"]
    text = f"""# External audit status

This page is the Paper 3 status. It does not replace the epoch-46 P128n128 narrative in the parent README.

The feasibility gate stopped the confirmatory study. Committed in-domain geologies: {len(decision['committed_geologies'])}. The blocked reasons are: {", ".join(decision["blocked_reasons"])}. The registered paper is {decision["paper"]}. Seed 0 remains the designated reported model, and its status is {decision["reported_model"]}. Seeds 1 and 2 are variability checks and were not trained.

The learning curve was {decision["learning_curve"]["status"]} and is labeled {decision["learning_curve"]["label"]}. That label does not classify dipping, three-layer, or realistic sections. Length scaling is {decision["length_scaling"]} because {decision["length_scaling_gate"]["n_pairs"]} finite-element pairs were available.

On IID training runs held out of the basis fit, the 128-mode output POD floor is {bands["overall"]:.6f} overall, {bands["low"]:.6f} below 0.4 Hz, {bands["mid"]:.6f} in 0.4–2.5 Hz, and {bands["high"]:.6f} in 2.5–10 Hz. The test split was not opened. The weighted input POD reconstruction is {floors["input_weighted_pod_reconstruction"]:.6f} at rank {floors["input_weighted_pod_rank"]}; the unweighted 16-component Vs PCA reconstruction is {floors["input_pca16_vs_reconstruction"]:.6f}. These are representation floors, not network scores.

No external label was decrypted. Treasure Island stays previously examined. Recorded motions stay descriptive and have no pass/fail gate.
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(OUT)


if __name__ == "__main__":
    main()
