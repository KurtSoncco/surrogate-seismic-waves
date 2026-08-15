#!/usr/bin/env python3
"""Inventory Box (or local) OOD corpora: tree, H5 count, TF cache, sample attrs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import config
from ood_io import default_ood_roots, probe_corpus


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Write probe dict to this path (in addition to stdout).",
    )
    p.add_argument(
        "--dipping",
        type=Path,
        default=None,
        help="Override ood_dipping root (else GIFNO_OOD_DIPPING / $GIFNO_DATA_ROOT/ood_dipping).",
    )
    p.add_argument(
        "--three-layer",
        type=Path,
        default=None,
        help="Override ood_three_layer root.",
    )
    args = p.parse_args()

    roots = default_ood_roots()
    if args.dipping is not None:
        roots["ood_dipping"] = args.dipping
    if args.three_layer is not None:
        roots["ood_three_layer"] = args.three_layer

    report = {
        "GIFNO_DATA_ROOT": str(config.data_root()),
        "corpora": {},
    }
    print(f"GIFNO_DATA_ROOT={config.data_root()}", flush=True)
    for name, root in roots.items():
        print(f"\n======== {name}  {root} ========", flush=True)
        info = probe_corpus(root)
        report["corpora"][name] = info
        for line in info["tree"]:
            print(line)
        print(f"n_h5={info['n_h5']}")
        print(f"tf_cache={info['tf_cache']}")
        print(f"root_manifest={info['root_manifest']}")
        sample = info.get("sample")
        if sample:
            print(f"sample={sample['path']}")
            print(f"  Vs_realization_2D={sample['Vs_realization_2D']}")
            print(f"  accel_n_channels={sample['accel_n_channels']}  shape={sample['accel_shape']}")
            print(f"  param_keys={sample['param_keys']}")
            print(f"  soil_nz={sample['soil_nz']}")
            print(f"  nominal={sample['nominal']}")
            interesting = [
                "Vs1",
                "Vs2",
                "H",
                "H_discretized",
                "CoV",
                "rf_seed",
                "dip_angle_deg",
                "dip_span",
                "dip_direction",
                "H1_discretized",
                "H2_discretized",
                "Vs_mid",
                "Vs_bedrock",
                "layer1_count",
                "layer2_count",
                "soil_layer_count",
            ]
            params = sample["params"]
            for k in interesting:
                if k in params:
                    print(f"    {k}={params[k]!r}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nWrote {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
