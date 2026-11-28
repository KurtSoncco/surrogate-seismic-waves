#!/usr/bin/env python3
"""Zone-wise Case 7 plus IID / dipping / three-layer (plan step 10).

    python run_eval.py --model-dir CK0 [--model-dir CK1] --out-dir results/eval_s0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

CASE = Path(__file__).resolve().parent
REPO = CASE.parents[1]
MIX_HOLD = REPO / "data" / "gifno_screen" / "outcrop_tf" / "mix_pretrain" / "holdout"
PY = REPO / ".venv" / "bin" / "python"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, action="append", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-per-campaign", type=int, default=48)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    md = [str(d) for d in args.model_dir for _ in [None]]
    flags = sum((["--model-dir", str(d)] for d in args.model_dir), [])
    subprocess.check_call(
        [str(PY), str(CASE / "score_section.py"), *flags, "--out-dir", str(args.out_dir / "case7")],
        cwd=str(CASE),
    )
    camp = args.out_dir / "campaigns.json"
    if MIX_HOLD.is_dir():
        subprocess.check_call(
            [str(PY), str(CASE / "score_campaigns.py"), "--data", str(MIX_HOLD), *flags,
             "--label", "model", "--max-per-campaign", str(args.max_per_campaign), "--out", str(camp)],
            cwd=str(CASE),
        )
    case7 = json.loads((args.out_dir / "case7" / "metrics.json").read_text())
    campaigns = json.loads(camp.read_text()) if camp.is_file() else {}
    summary = {"case7": {k: case7[k] for k in ("tf1d_prior", "center", "overlap", "all21") if k in case7}, "campaigns": campaigns.get("campaigns", {})}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2)[:2000])


if __name__ == "__main__":
    main()
