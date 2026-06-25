"""Standalone speed/accuracy profiler for the LOGLO-POD model (no wandb).

Reports, on real data:
  1. fp32-vs-bf16 output equivalence (AMP accuracy safety),
  2. per-component forward timing (lift / encoder / head),
  3. top CUDA ops (where the time goes),
  4. batch-size scaling: ms/it, samples/s, peak GB for fp32 and bf16.

Usage (local or Delta):
    python profile_speed.py --limit 500 --batches 1,2,4,8,16,32
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile

import config

config.setup_import_paths()
from data_loader import get_data_loaders  # noqa: E402
from losses import build_training_loss  # noqa: E402
from model import create_model  # noqa: E402


def ensure_pod(limit: int) -> None:
    if config.POD_MODES_PATH.exists() and config.POD_MEAN_PATH.exists():
        return
    config.POD_MODES_PATH.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(config.POD_PREPROCESS_SCRIPT),
            "--limit",
            str(limit),
            "--n-modes",
            str(config.POD_NUM_MODES),
            "--seed",
            str(config.SEED),
            "--train-split",
            str(config.TRAIN_SPLIT),
            "--val-split",
            str(config.VAL_SPLIT),
            "--out-modes",
            str(config.POD_MODES_PATH),
            "--out-mean",
            str(config.POD_MEAN_PATH),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--batches", type=str, default="1,2,4,8,16,32")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--compile", action="store_true", help="torch.compile the model")
    parser.add_argument("--latent", type=int, default=None,
                        help="override LATENT_CHANNELS (capacity-vs-speed test)")
    parser.add_argument("--layers", type=int, default=None,
                        help="override NUM_FNO_LAYERS")
    args = parser.parse_args()
    batches = [int(b) for b in args.batches.split(",") if b.strip()]
    pool = max(batches)

    dev = config.DEVICE
    cuda = dev.type == "cuda"
    name = torch.cuda.get_device_name(0) if cuda else "cpu"
    print("=" * 70)
    print(
        f"device={dev} gpu={name} torch={torch.__version__} "
        f"bf16={cuda and torch.cuda.is_bf16_supported()}"
    )
    if cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    ensure_pod(args.limit)
    tr, _, _, _ = get_data_loaders(limit=args.limit, batch_size=pool, num_workers=0)
    x8, y8, m8 = next(iter(tr))
    x8, y8, m8 = x8.to(dev), y8.to(dev), m8.to(dev)

    model_kwargs = {}
    if args.latent is not None:
        model_kwargs["latent_channels"] = args.latent
    if args.layers is not None:
        model_kwargs["num_fno_layers"] = args.layers
    raw_model = create_model(**model_kwargs).to(dev)
    model = raw_model
    if args.compile:
        # Inductor has no complex64 Triton codegen; let complex subgraphs fall
        # back to eager while still fusing the non-complex elementwise regions.
        torch._dynamo.config.suppress_errors = True
        try:
            compiled = torch.compile(raw_model)
            with torch.no_grad(), torch.autocast(
                device_type=dev.type, dtype=torch.bfloat16, enabled=cuda
            ):
                compiled(x8[:1])  # trigger compilation now to surface failures
            if cuda:
                torch.cuda.synchronize()
            model = compiled
            print("[compile] torch.compile OK")
        except Exception as e:  # noqa: BLE001
            print(f"[compile] FAILED ({type(e).__name__}: {e}); falling back to eager")
            model = raw_model
    crit = build_training_loss()
    n_params = sum(p.numel() for p in raw_model.parameters())
    print(
        f"params={n_params / 1e6:.2f}M  input={tuple(x8.shape[1:])}  "
        f"target={tuple(y8.shape[1:])}  pooled_batch={x8.shape[0]}"
    )
    print(
        f"compile={args.compile}  latent={args.latent or config.LATENT_CHANNELS}  "
        f"layers={args.layers or config.NUM_FNO_LAYERS}"
    )
    print("=" * 70)

    def amp_ctx(enabled: bool):
        return torch.autocast(
            device_type=dev.type, dtype=torch.bfloat16, enabled=enabled
        )

    def sync():
        if cuda:
            torch.cuda.synchronize()

    # (1) accuracy
    print("\n=== accuracy (fp32 vs bf16 output) ===")
    model.eval()
    acc_b = min(2, pool)
    with torch.no_grad():
        xb = x8[:acc_b]
        o32 = model(xb)
        with amp_ctx(True):
            obf = model(xb)
        rel = (obf.float() - o32).norm() / (o32.norm() + 1e-12)
        print(
            f"  bs={acc_b}: rel-diff={rel.item():.3e} "
            f"finite={torch.isfinite(obf).all().item()}"
        )

    # (2) per-component forward timing
    comp_b = min(4, pool)
    print(f"\n=== per-component forward timing (bs={comp_b}, bf16, uncompiled) ===")
    raw_model.eval()
    xb = x8[:comp_b]
    with torch.no_grad(), amp_ctx(True):
        lifted = raw_model.lift(xb)
        xg, xl = raw_model.encoder(lifted)

    def t_fwd(fn, iters=10, warmup=3):
        with torch.no_grad(), amp_ctx(True):
            for _ in range(warmup):
                fn()
            sync()
            t0 = time.time()
            for _ in range(iters):
                fn()
            sync()
        return (time.time() - t0) / iters * 1000.0

    t_lift = t_fwd(lambda: raw_model.lift(xb))
    t_enc = t_fwd(lambda: raw_model.encoder(lifted))
    t_head = t_fwd(lambda: raw_model.head(xg, xl))
    print(f"  lift={t_lift:.1f}ms  encoder={t_enc:.1f}ms  head={t_head:.1f}ms")

    # (3) op-level profiler
    if cuda:
        print(f"\n=== top CUDA ops (bs={comp_b}, bf16, fwd+bwd) ===")
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        xb, yb, mb = x8[:comp_b], y8[:comp_b], m8[:comp_b]
        for _ in range(3):
            opt.zero_grad()
            with amp_ctx(True):
                loss = crit(model(xb), yb, mb)
            loss.backward()
            opt.step()
        sync()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(5):
                opt.zero_grad()
                with amp_ctx(True):
                    loss = crit(model(xb), yb, mb)
                loss.backward()
                opt.step()
            sync()
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))

    # (4) batch-size scaling
    def bench(b: int, amp: bool):
        xb, yb, mb = x8[:b], y8[:b], m8[:b]
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        for _ in range(args.warmup):
            opt.zero_grad()
            with amp_ctx(amp):
                loss = crit(model(xb), yb, mb)
            loss.backward()
            opt.step()
        sync()
        t0 = time.time()
        for _ in range(args.iters):
            opt.zero_grad()
            with amp_ctx(amp):
                loss = crit(model(xb), yb, mb)
            loss.backward()
            opt.step()
        sync()
        ms = (time.time() - t0) / args.iters * 1000.0
        peak = torch.cuda.max_memory_allocated() / 1e9 if cuda else 0.0
        return ms, peak

    print("\n=== batch-size scaling (fwd+bwd) ===")
    print(f"{'bs':>4} {'mode':>5} {'ms/it':>9} {'samp/s':>9} {'peakGB':>8}")
    model.train()
    best = (None, 0.0)
    for amp in (False, True):
        for b in batches:
            if b > pool:
                continue
            try:
                ms, peak = bench(b, amp)
                sps = b / ms * 1000.0
                tag = "bf16" if amp else "fp32"
                print(f"{b:>4} {tag:>5} {ms:>9.1f} {sps:>9.1f} {peak:>8.2f}")
                if sps > best[1]:
                    best = (f"bs={b} {tag}", sps)
                if cuda:
                    torch.cuda.empty_cache()
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                print(f"{b:>4} {'bf16' if amp else 'fp32':>5}   OOM/{type(e).__name__}")
                if cuda:
                    torch.cuda.empty_cache()
                break  # larger batches will also fail

    if best[0]:
        print(f"\nbest throughput: {best[0]} -> {best[1]:.1f} samples/s")
    print("\nOK")


if __name__ == "__main__":
    main()
