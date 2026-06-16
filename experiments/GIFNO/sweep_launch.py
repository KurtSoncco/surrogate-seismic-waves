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


def load_variants(path: Path) -> list[SweepVariant]:
    variants: list[SweepVariant] = []
    for line in path.read_text().splitlines():
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


def build_export_env(
    variant: SweepVariant,
    tf_dir: Path,
    *,
    screen: bool,
) -> dict[str, str]:
    suffix = "screen" if screen else "full"
    env = {
        "WANDB_RUN_NAME": f"sweep_{variant.name}_{suffix}",
        "GIFNO_MODEL_DIR": str(tf_dir / "models" / "sweep" / variant.name),
        "GIFNO_RESULTS_DIR": str(tf_dir / "results" / "sweep" / variant.name),
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
    env.update(build_export_env(variant, tf_dir, screen=screen))

    cmd = ["sbatch", f"--job-name=gifno_{variant.name}", "--export=ALL"]
    train_script = gifno_dir / "delta_train.sh"
    main_args: list[str] = []
    if limit is not None:
        main_args.extend(["--limit", str(limit)])
    cmd.append(str(train_script))
    cmd.extend(main_args)

    override_str = ", ".join(f"{k}={v}" for k, v in variant.overrides.items()) or "(baseline)"
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
    parser = argparse.ArgumentParser(description="Submit GIFNO hyperparameter sweep jobs")
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
