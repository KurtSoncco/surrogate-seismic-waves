"""IID residual scale runs: cache_tag × encoder × n_freq × seed."""

from __future__ import annotations

import argparse
import json

import config
from train import train_one


def main() -> None:
    p = argparse.ArgumentParser(description="DeepONet-Residual scale / freq runs")
    p.add_argument("--cache-tag", default="n2000_seed42")
    p.add_argument("--target", choices=["R_col", "R_nom"], default="R_nom")
    p.add_argument(
        "--branch-mode",
        choices=["single", "multi", "stoch_only", "fields_only"],
        default="single",
    )
    p.add_argument(
        "--trunk-set",
        choices=["fstar", "fstar_fourier", "xL", "full"],
        default="full",
    )
    p.add_argument(
        "--field-encoder",
        choices=["conv", "resunet"],
        default="resunet",
    )
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--patience", type=int, default=60)
    p.add_argument("--n-freq-train", type=int, default=config.N_FREQ_TRAIN)
    p.add_argument("--n-freq-eval", type=int, default=config.N_FREQ_EVAL)
    p.add_argument("--no-early-stop", action="store_true")
    args = p.parse_args()
    result = train_one(
        cache_tag=args.cache_tag,
        target=args.target,  # type: ignore[arg-type]
        branch_mode=args.branch_mode,  # type: ignore[arg-type]
        trunk_set=args.trunk_set,  # type: ignore[arg-type]
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        patience=args.patience,
        no_early_stop=args.no_early_stop,
        field_encoder=args.field_encoder,  # type: ignore[arg-type]
        n_freq_train=args.n_freq_train,
        n_freq_eval=args.n_freq_eval,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
