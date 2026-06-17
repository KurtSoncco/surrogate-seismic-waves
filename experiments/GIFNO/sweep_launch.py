#!/usr/bin/env python3
"""Parse sweep_variants.tsv and submit SLURM training jobs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SweepVariant:
    name: str
    overrides: dict[str, str] = field(default_factory=dict)


# Built-in fallback if sweep_variants.tsv was not synced to the cluster.
DEFAULT_VARIANTS_TSV = """\
baseline
h1_strong\tLOSS_H1_WEIGHT=0.25
fno_wide\tFNO_MODES=48,48
latent_wide\tLATENT_CHANNELS=96
freq_loss\tLOSS_FREQ_WEIGHT=0.05
no_mining\tHARD_MINING=false
"""

DEFAULT_VARIANTS_R2_TSV = """\
lw_anchor\tLATENT_CHANNELS=96
lw_no_mine\tLATENT_CHANNELS=96;HARD_MINING=false
lw_fno\tLATENT_CHANNELS=96;FNO_MODES=48,48
lw_no_mine_fno\tLATENT_CHANNELS=96;HARD_MINING=false;FNO_MODES=48,48
lw_no_mine_h1\tLATENT_CHANNELS=96;HARD_MINING=false;LOSS_H1_WEIGHT=0.25
lw_no_mine_freq\tLATENT_CHANNELS=96;HARD_MINING=false;LOSS_FREQ_WEIGHT=0.05
lw_anchor_lr5e3\tLATENT_CHANNELS=96;LEARNING_RATE=0.005
lw_no_mine_lr5e3\tLATENT_CHANNELS=96;HARD_MINING=false;LEARNING_RATE=0.005
lw_no_mine_fno_lr5e3\tLATENT_CHANNELS=96;HARD_MINING=false;FNO_MODES=48,48;LEARNING_RATE=0.005
"""

DEFAULT_VARIANTS_R3_TSV = """\
lw_nm_logtf\tLATENT_CHANNELS=96;HARD_MINING=false;LOG_TF_LOSS=true
lw_nm_linf\tLATENT_CHANNELS=96;HARD_MINING=false;LOSS_LINF_WEIGHT=0.1
lw_nm_valley\tLATENT_CHANNELS=96;HARD_MINING=false;VALLEY_LOSS_WEIGHT=2.0
lw_nm_h1\tLATENT_CHANNELS=96;HARD_MINING=false;LOSS_H1_WEIGHT=0.25
"""


def load_variants_from_text(text: str) -> list[SweepVariant]:
    variants: list[SweepVariant] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            name, overrides_raw = line.split("\t", 1)
            overrides: dict[str, str] = {}
            for pair in overrides_raw.split(";"):
                pair = pair.strip()
                if not pair:
                    continue
                key, val = pair.split("=", 1)
                overrides[key.strip()] = val.strip()
        else:
            name, overrides = line, {}
        variants.append(SweepVariant(name=name.strip(), overrides=overrides))
    return variants


def load_variants(path: Path) -> list[SweepVariant]:
    if path.is_file():
        return load_variants_from_text(path.read_text())
    if path.name == "sweep_variants_r3.tsv":
        fallback = DEFAULT_VARIANTS_R3_TSV
    elif path.name == "sweep_variants_r2.tsv":
        fallback = DEFAULT_VARIANTS_R2_TSV
    else:
        fallback = DEFAULT_VARIANTS_TSV
    print(
        f"[GIFNO sweep] WARNING: {path} not found — using built-in fallback.",
        file=sys.stderr,
        flush=True,
    )
    return load_variants_from_text(fallback)


def sweep_run_tag(*, screen: bool, limit: int | None) -> str:
    """Unique tag for W&B and output dirs (e.g. n1000, n4000, full)."""
    if not screen:
        return "full"
    return f"n{limit}" if limit is not None else "screen"


def build_export_env(
    variant: SweepVariant,
    tf_dir: Path,
    *,
    screen: bool,
    limit: int | None,
) -> dict[str, str]:
    tag = sweep_run_tag(screen=screen, limit=limit)
    sweep_root = "sweep" if tag == "full" else f"sweep/{tag}"
    env = {
        "WANDB_RUN_NAME": f"sweep_{variant.name}_{tag}",
        "GIFNO_MODEL_DIR": str(tf_dir / "models" / sweep_root / variant.name),
        "GIFNO_RESULTS_DIR": str(tf_dir / "results" / sweep_root / variant.name),
    }
    for key, val in variant.overrides.items():
        env[f"GIFNO_{key}"] = val
    return env


def submit_job(
    variant: SweepVariant,
    gifno_dir: Path,
    tf_dir: Path,
    *,
    screen: bool,
    limit: int | None,
    dry_run: bool,
) -> str | None:
    env = os.environ.copy()
    env.update(build_export_env(variant, tf_dir, screen=screen, limit=limit))

    cmd = ["sbatch", f"--job-name=gifno_{variant.name}", "--export=ALL"]
    train_script = gifno_dir / "delta_train.sh"
    main_args: list[str] = []
    if limit is not None:
        main_args.extend(["--limit", str(limit)])
    cmd.append(str(train_script))
    cmd.extend(main_args)

    override_str = (
        ", ".join(f"{k}={v}" for k, v in variant.overrides.items()) or "(baseline)"
    )
    print(f"  variant={variant.name}  overrides={override_str}")
    print(f"    WANDB_RUN_NAME={env['WANDB_RUN_NAME']}")
    print(f"    GIFNO_MODEL_DIR={env['GIFNO_MODEL_DIR']}")

    if dry_run:
        print(f"    [dry-run] would run: {' '.join(cmd)}")
        return None

    result = subprocess.run(cmd, env=env, cwd=gifno_dir, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr or result.stdout, file=sys.stderr)
        raise RuntimeError(f"sbatch failed for {variant.name}")
    job_id = result.stdout.strip().split()[-1]
    print(f"    submitted job {job_id}")
    return job_id


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Submit GIFNO hyperparameter sweep jobs"
    )
    parser.add_argument(
        "--variants",
        type=Path,
        default=Path(__file__).resolve().parent / "sweep_variants.tsv",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full dataset (no --limit); use after screening",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Dataset limit for screening runs (default: 1000)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Submit only this variant (for rerun)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    gifno_dir = Path(__file__).resolve().parent
    tf_dir = Path(os.environ.get("GIFNO_TF_DIR", ""))
    if not tf_dir:
        data_root = os.environ.get("GIFNO_DATA_ROOT", "")
        if data_root:
            tf_dir = Path(data_root) / "transfer_function"

    variants = load_variants(args.variants)
    if args.name:
        variants = [v for v in variants if v.name == args.name]
        if not variants:
            raise SystemExit(f"Unknown variant: {args.name}")

    limit = None if args.full else args.limit
    mode = "full" if args.full else f"screen (limit={limit})"
    print(f"=== GIFNO sweep: {len(variants)} job(s), mode={mode} ===")

    job_ids: list[str] = []
    for variant in variants:
        jid = submit_job(
            variant,
            gifno_dir,
            tf_dir,
            screen=not args.full,
            limit=limit,
            dry_run=args.dry_run,
        )
        if jid:
            job_ids.append(jid)

    print("")
    print("Submitted:", ", ".join(job_ids) if job_ids else "(dry-run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
