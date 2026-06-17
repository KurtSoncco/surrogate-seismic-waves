#!/usr/bin/env python3
"""Submit GIFNO-FDO sweep jobs to SLURM."""

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


DEFAULT_VARIANTS_ARCH_TSV = """\
fdn_h1
fdn_h1_wide\tDEEPONET_LATENT_DIM=128
fdn_h1_depth\tBRANCH_MODE=depth
"""

DEFAULT_VARIANTS_COMBO_TSV = """\
lw_nm_h1_valley\tLOSS_H1_WEIGHT=0.25;VALLEY_LOSS_WEIGHT=2.0
"""

DEFAULT_VARIANTS_OPT_TSV = """\
wide_h1_p2_adam\tDEEPONET_LATENT_DIM=128
wide_h1_p1_adam\tDEEPONET_LATENT_DIM=128;LOSS_P=1
wide_h1_p2_adamw\tDEEPONET_LATENT_DIM=128;OPTIMIZER=adamw
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
    fallback = DEFAULT_VARIANTS_COMBO_TSV
    if path.name == "sweep_variants_arch.tsv":
        fallback = DEFAULT_VARIANTS_ARCH_TSV
    elif path.name == "sweep_variants_opt.tsv":
        fallback = DEFAULT_VARIANTS_OPT_TSV
    elif path.name != "sweep_variants_combo.tsv":
        fallback = DEFAULT_VARIANTS_ARCH_TSV
    print(
        f"[GIFNO-FDO sweep] WARNING: {path} not found — using built-in fallback.",
        file=sys.stderr,
        flush=True,
    )
    return load_variants_from_text(fallback)


def sweep_run_tag(*, screen: bool, limit: int | None) -> str:
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
    sweep_root = "fdo/sweep" if tag == "full" else f"fdo/sweep/{tag}"
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
    fdo_dir: Path,
    tf_dir: Path,
    *,
    screen: bool,
    limit: int | None,
    dry_run: bool,
) -> str | None:
    env = os.environ.copy()
    env.update(build_export_env(variant, tf_dir, screen=screen, limit=limit))

    cmd = ["sbatch", f"--job-name=fdo_{variant.name}", "--export=ALL"]
    train_script = fdo_dir / "delta_train.sh"
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

    result = subprocess.run(cmd, env=env, cwd=fdo_dir, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr or result.stdout, file=sys.stderr)
        raise RuntimeError(f"sbatch failed for {variant.name}")
    job_id = result.stdout.strip().split()[-1]
    print(f"    submitted job {job_id}")
    return job_id


def main() -> int:
    parser = argparse.ArgumentParser(description="Submit GIFNO-FDO sweep jobs")
    parser.add_argument(
        "--variants",
        type=Path,
        default=Path(__file__).resolve().parent / "sweep_variants_arch.tsv",
    )
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    fdo_dir = Path(__file__).resolve().parent
    variants_path = args.variants
    if not variants_path.is_absolute():
        variants_path = fdo_dir / variants_path
    tf_dir = Path(os.environ.get("GIFNO_TF_DIR", ""))
    if not tf_dir:
        data_root = os.environ.get("GIFNO_DATA_ROOT", "")
        if data_root:
            tf_dir = Path(data_root) / "transfer_function"

    variants = load_variants(variants_path)
    if args.name:
        variants = [v for v in variants if v.name == args.name]
        if not variants:
            raise SystemExit(f"Unknown variant: {args.name}")

    limit = None if args.full else args.limit
    mode = "full" if args.full else f"screen (limit={limit})"
    print(f"=== GIFNO-FDO sweep: {len(variants)} job(s), mode={mode} ===")

    job_ids: list[str] = []
    for variant in variants:
        jid = submit_job(
            variant,
            fdo_dir,
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
