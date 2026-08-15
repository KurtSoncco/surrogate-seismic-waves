#!/usr/bin/env python3
"""IID scale ladder: cache_tag × encoder × n_freq_train × seed."""

from __future__ import annotations

import argparse
import json

import config
from train import train_one


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-tag", default="n2000_seed42")
    p.add_argument("--encoder", choices=["conv", "resunet"], default=config.DEFAULT_FIELD_ENCODER)
    p.add_argument("--n-freq-train", type=int, default=config.N_FREQ_TRAIN)
    p.add_argument("--n-freq-eval", type=int, default=config.N_FREQ_EVAL)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--target", choices=["R_col", "R_nom"], default=config.DEFAULT_TARGET)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--patience", type=int, default=config.PATIENCE)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--no-early-stop", action="store_true")
    p.add_argument(
        "--serial-tf1d",
        action=argparse.BooleanOptionalAction,
        default=config.DEFAULT_SERIAL_TF1D,
    )
    args = p.parse_args()

    result = train_one(
        cache_tag=args.cache_tag,
        target=args.target,  # type: ignore[arg-type]
        branch_mode="single",
        trunk_set="full",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        patience=args.patience,
        no_early_stop=args.no_early_stop,
        field_encoder=args.encoder,  # type: ignore[arg-type]
        n_freq_train=args.n_freq_train,
        n_freq_eval=args.n_freq_eval,
        serial_tf1d=args.serial_tf1d,
        run_name=(
            f"single_{args.encoder}_full_{args.target}_{args.cache_tag}"
            f"_nf{args.n_freq_train}_seed{args.seed}"
            f"{'_serial' if args.serial_tf1d else ''}"
        ),
    )
    out = config.RESULTS_DIR / f"scale_{result['name']}.json"
    out.write_text(json.dumps(result, indent=2))
    test = result["test"]
    print(
        f"\n[scale] {result['name']}\n"
        f"  r2_R={test['r2_R']:.3f}  pearson_R_freq={test['pearson_R_freq']:.3f}  "
        f"delta_r2_TF={test['delta_r2_TF']:+.3f}  "
        f"(n_freq_train={args.n_freq_train}, n_freq_eval={args.n_freq_eval})",
        flush=True,
    )
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
