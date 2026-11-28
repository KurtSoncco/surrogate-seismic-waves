#!/usr/bin/env python3
"""Stage 0 ablation, the pair-weight pick, three seeds, then stages 1 and 2.

Designed for one A10. A machine without CUDA runs only when --allow-cpu is set,
and then only the smoke path (two epochs). The real run refuses CPU.

    python curriculum.py --cache ../../data/vardomain/cache --out ../../checkpoints/vardomain
    python curriculum.py --smoke --allow-cpu --cache ... --out /tmp/vardomain_smoke
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
TRAIN = HERE / "train.py"
SCORE = HERE / "score_holdout.py"


def _run(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def _val(path: Path) -> float:
    hist = path / "history.csv"
    if not hist.is_file():
        return float("inf")
    lines = hist.read_text().strip().splitlines()[1:]
    scores = []
    for row in lines:
        parts = row.split(",")
        if len(parts) < 3:
            continue
        # agreement is column 8. An epoch the checkpoint would reject does not choose the stage.
        if len(parts) > 8 and not (float(parts[8]) < 0.0136):
            continue
        scores.append(float(parts[2]))
    return min(scores) if scores else float("inf")


def _seed_ratio(path: Path) -> float | None:
    ev = path / "eval_test.json"
    if not ev.is_file():
        return None
    data = json.loads(ev.read_text())
    return data.get("seed", {}).get("sigma_ratio_mid")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--resume", action="store_true", help="skip a stage whose eval_test.json is already written")
    ap.add_argument("--jitter", action="store_true")
    ap.add_argument("--whiten", action="store_true")
    ap.add_argument("--curvature", type=float, default=0.0)
    ap.add_argument("--hfp-kernel", type=int, default=4)
    ap.add_argument("--hfp-stride", type=int, default=4)
    ap.add_argument("--local-min-wavelength", type=float, default=0.0)
    args = ap.parse_args()
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("this curriculum is for one CUDA GPU (an A10). Pass --allow-cpu only for a smoke run.")
    if not torch.cuda.is_available() and not args.smoke:
        raise SystemExit("refusing a full curriculum on CPU")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    py = args.python
    smoke = ["--smoke"] if args.smoke else []

    lateral = [
        "--hfp-kernel", str(args.hfp_kernel),
        "--hfp-stride", str(args.hfp_stride),
        "--local-min-wavelength", str(args.local_min_wavelength),
        "--curvature", str(args.curvature),
    ]
    if args.jitter:
        lateral.append("--jitter")
    if args.whiten:
        lateral.append("--whiten")

    def train(*extra: str) -> None:
        dest = Path(extra[extra.index("--out") + 1]) if "--out" in extra else None
        if args.resume and dest is not None and (dest / "eval_test.json").is_file():
            print(f"resume skip {dest}", flush=True)
            return
        _run([py, str(TRAIN), "--cache", str(args.cache), "--encoding", "metric", *smoke, *lateral, *extra])

    # Stage 0: one residual seed and one direct seed.
    res_dir = out / "stage0_residual_s0"
    dir_dir = out / "stage0_direct_s0"
    train("--stage", "0", "--target", "residual", "--seed", "0", "--out", str(res_dir), "--overwrite")
    train("--stage", "0", "--target", "direct", "--seed", "0", "--out", str(dir_dir), "--overwrite")
    winner_target = "residual" if _val(res_dir) <= _val(dir_dir) else "direct"
    winner = res_dir if winner_target == "residual" else dir_dir

    # Pair weight 0 is the ablation above. The 0.25 sweep is shorter; the three
    # seeds below are the equal-epoch retrain and are the ones the gate uses.
    pair_epochs = ["--epochs", "2" if args.smoke else "40"]
    pair_dir = out / f"stage0_{winner_target}_pair_s0"
    train("--stage", "0", "--target", winner_target, "--pair-weight", "0.25", "--seed", "0",
          "--out", str(pair_dir), "--overwrite", *pair_epochs)
    r0, r1 = _seed_ratio(winner), _seed_ratio(pair_dir)
    pair_weight = "0"
    if r0 is not None and r1 is not None and abs(r1 - 1.0) < abs(r0 - 1.0):
        pair_weight = "0.25"
    (out / "stage0_choice.json").write_text(json.dumps({
        "target": winner_target, "pair_weight": float(pair_weight),
        "residual_val": _val(res_dir), "direct_val": _val(dir_dir),
        "sigma_ratio_no_pair": r0, "sigma_ratio_pair": r1,
    }, indent=2))

    seeds = (0,) if args.smoke else (0, 1, 2)
    stage0_dirs = []
    for seed in seeds:
        dest = out / f"stage0_{winner_target}_s{seed}"
        # seed 0 at pair weight 0 is already a full ablation run
        if seed == 0 and pair_weight == "0":
            stage0_dirs.append(winner)
            continue
        train("--stage", "0", "--target", winner_target, "--pair-weight", pair_weight, "--seed", str(seed),
              "--out", str(dest), "--overwrite")
        stage0_dirs.append(dest)
    ratios, explained, vals = [], [], []
    for d in stage0_dirs:
        ev = json.loads((d / "eval_test.json").read_text()) if (d / "eval_test.json").is_file() else {}
        seed = ev.get("seed") or {}
        ratios.append(seed.get("sigma_ratio_mid"))
        explained.append(seed.get("seed_explained_mid"))
        vals.append(_val(d))
    finite = [r for r in ratios if r is not None]
    gate = {
        "sigma_ratio_mid": ratios,
        "sigma_ratio_spread": (max(finite) - min(finite)) if len(finite) >= 2 else None,
        "seed_explained_mid": explained,
        "val": vals,
        "sigma_in_0.9_1.1": bool(finite) and all(0.9 <= r <= 1.1 for r in finite),
        "explained_above_0.17": bool(explained) and all(e is not None and e > 0.17 for e in explained),
    }
    (out / "GATE.json").write_text(json.dumps(gate, indent=2))
    print(json.dumps({"gate": gate}, indent=2), flush=True)
    best0 = min(stage0_dirs, key=_val)

    stage1 = out / "stage1"
    train("--stage", "1", "--target", winner_target, "--pair-weight", pair_weight, "--seed", "0",
          "--init", str(best0), "--out", str(stage1), "--overwrite")
    iid_base = _val(stage1)
    # stage 1's history val_score is the IID score (selection uses iid bins)
    stage2 = out / "stage2"
    train("--stage", "2", "--target", winner_target, "--pair-weight", pair_weight, "--seed", "0",
          "--init", str(stage1), "--iid-baseline", f"{iid_base}", "--out", str(stage2), "--overwrite")
    _run([py, str(SCORE), "--model-dir", str(stage2), "--cache", str(args.cache)])
    print(f"stage 2 checkpoint {stage2}", flush=True)


if __name__ == "__main__":
    main()
