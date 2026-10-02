#!/usr/bin/env python3
"""Submit GIFNO-FDO-XT-LOGLO-POD sweep jobs to SLURM."""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path


def _load_gifno_sweep_launch():
    """Parser and run-tag helpers are shared with ``experiments/GIFNO/sweep_launch.py``."""
    path = Path(__file__).resolve().parents[1] / "GIFNO" / "sweep_launch.py"
    spec = importlib.util.spec_from_file_location("gifno_sweep_launch", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gifno_sweep_launch"] = mod
    spec.loader.exec_module(mod)
    return mod


_gifno_sweep = _load_gifno_sweep_launch()
SweepVariant = _gifno_sweep.SweepVariant
load_variants_from_text = _gifno_sweep.load_variants_from_text
sweep_run_tag = _gifno_sweep.sweep_run_tag


DEFAULT_VARIANTS_LOGLO_POD_TSV = """\
loglo_pod_ref\tLATENT_CHANNELS=128;POD_NUM_MODES=32;LOSS_RADIAL_WEIGHT=0.25;LOGLO_PATCH_SIZE=16,20;LOGLO_HF_NOISE_ALPHA=0.025
loglo_pod_pod48\tLATENT_CHANNELS=128;POD_NUM_MODES=48;LOSS_RADIAL_WEIGHT=0.25;LOGLO_PATCH_SIZE=16,20;LOGLO_HF_NOISE_ALPHA=0.025
loglo_pod_pod64\tLATENT_CHANNELS=128;POD_NUM_MODES=64;LOSS_RADIAL_WEIGHT=0.25;LOGLO_PATCH_SIZE=16,20;LOGLO_HF_NOISE_ALPHA=0.025
loglo_pod_no_radial\tLATENT_CHANNELS=128;POD_NUM_MODES=32;LOSS_RADIAL_WEIGHT=0.0;LOGLO_PATCH_SIZE=16,20;LOGLO_HF_NOISE_ALPHA=0.025
loglo_pod_patchw10\tLATENT_CHANNELS=128;POD_NUM_MODES=32;LOSS_RADIAL_WEIGHT=0.25;LOGLO_PATCH_SIZE=16,10;LOGLO_HF_NOISE_ALPHA=0.025
loglo_pod_no_hfnoise\tLATENT_CHANNELS=128;POD_NUM_MODES=32;LOSS_RADIAL_WEIGHT=0.25;LOGLO_PATCH_SIZE=16,20;LOGLO_HF_NOISE_ALPHA=0.0
loglo_pod_depth32\tLATENT_CHANNELS=128;POD_NUM_MODES=32;LOSS_RADIAL_WEIGHT=0.25;LOGLO_DEPTH_STRIDE=4;LOGLO_PATCH_SIZE=16,20;LOGLO_HF_NOISE_ALPHA=0.025
loglo_pod_depth_full\tLATENT_CHANNELS=128;POD_NUM_MODES=32;LOSS_RADIAL_WEIGHT=0.25;LOGLO_DEPTH_STRIDE=1;LOGLO_PATCH_SIZE=16,20;LOGLO_HF_NOISE_ALPHA=0.025
loglo_pod_latent256\tLATENT_CHANNELS=256;POD_NUM_MODES=32;LOSS_RADIAL_WEIGHT=0.25;LOGLO_PATCH_SIZE=16,20;LOGLO_HF_NOISE_ALPHA=0.025
"""


def load_variants(path: Path) -> list[SweepVariant]:
    if path.is_file():
        return load_variants_from_text(path.read_text())
    print(
        f"[GIFNO-FDO-XT-LOGLO-POD sweep] WARNING: {path} not found — using built-in fallback.",
        file=sys.stderr,
        flush=True,
    )
    return load_variants_from_text(DEFAULT_VARIANTS_LOGLO_POD_TSV)


def build_export_env(
    variant: SweepVariant,
    tf_dir: Path,
    *,
    screen: bool,
    limit: int | None,
) -> dict[str, str]:
    tag = sweep_run_tag(screen=screen, limit=limit)
    sweep_root = (
        "fdo_xt_loglo_pod/sweep" if tag == "full" else f"fdo_xt_loglo_pod/sweep/{tag}"
    )
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
    xt_dir: Path,
    tf_dir: Path,
    *,
    screen: bool,
    limit: int | None,
    walltime: str,
    dry_run: bool,
) -> str | None:
    env = os.environ.copy()
    env.update(build_export_env(variant, tf_dir, screen=screen, limit=limit))

    cmd = ["sbatch", f"--job-name={variant.name}", "--export=ALL"]
    # sbatch CLI --time overrides delta_train.sh's #SBATCH directive.
    cmd.append(f"--time={walltime}")
    train_script = xt_dir / "delta_train.sh"
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

    result = subprocess.run(cmd, env=env, cwd=xt_dir, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr or result.stdout, file=sys.stderr)
        raise RuntimeError(f"sbatch failed for {variant.name}")
    job_id = result.stdout.strip().split()[-1]
    print(f"    submitted job {job_id}")
    return job_id


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Submit GIFNO-FDO-XT-LOGLO-POD sweep jobs"
    )
    parser.add_argument(
        "--variants",
        type=Path,
        default=Path(__file__).resolve().parent / "sweep_variants_loglo_pod.tsv",
    )
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument(
        "--time",
        type=str,
        default=None,
        help="SLURM walltime for sbatch (default: 04:00:00 screen, 24:00:00 full)",
    )
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    xt_dir = Path(__file__).resolve().parent
    variants_path = args.variants
    if not variants_path.is_absolute():
        variants_path = xt_dir / variants_path
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
    walltime = args.time or ("24:00:00" if args.full else "04:00:00")
    mode = "full" if args.full else f"screen (limit={limit})"
    print(
        f"=== GIFNO-FDO-XT-LOGLO-POD sweep: {len(variants)} job(s), "
        f"mode={mode}, walltime={walltime} ==="
    )

    job_ids: list[str] = []
    for variant in variants:
        jid = submit_job(
            variant,
            xt_dir,
            tf_dir,
            screen=not args.full,
            limit=limit,
            walltime=walltime,
            dry_run=args.dry_run,
        )
        if jid:
            job_ids.append(jid)

    print("")
    print("Submitted:", ", ".join(job_ids) if job_ids else "(dry-run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
