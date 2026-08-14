"""Stratified GIFNO sample indices for residual screen packs (no RF screen)."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import config
import numpy as np

_RES = config.RESIDUAL_DIR
if str(_RES) not in sys.path:
    sys.path.insert(0, str(_RES))

# ruff: noqa: E402
from residual_target import load_manifest, stratified_sample_indices

_CACHE_TAG_RE = re.compile(r"^n(\d+)_seed(\d+)$")


def parse_cache_tag(tag: str) -> tuple[int, int]:
    m = _CACHE_TAG_RE.fullmatch(tag.strip())
    if not m:
        raise ValueError(f"cache tag must look like n2000_seed42, got {tag!r}")
    return int(m.group(1)), int(m.group(2))


def parent_cache_tag(seed: int, n_max: int = config.SCREEN_N_MAX) -> str:
    return f"n{int(n_max)}_seed{int(seed)}"


def _load_if_exists(path: Path) -> np.ndarray | None:
    if path.is_file():
        return np.load(path)
    return None


def _largest_nested_indices(n: int, seed: int) -> np.ndarray | None:
    """Largest cached draw with the same seed and size strictly below n."""
    best: np.ndarray | None = None
    best_k = -1
    for root in (config.RESIDUAL_CACHE_DIR, config.CACHE_DIR):
        if not root.is_dir():
            continue
        for child in root.iterdir():
            if not child.is_dir():
                continue
            try:
                k, s = parse_cache_tag(child.name)
            except ValueError:
                continue
            if s != seed or k >= n or k <= best_k:
                continue
            loaded = _load_if_exists(child / "sample_indices.npy")
            if loaded is None:
                continue
            best = np.asarray(loaded, dtype=int)
            best_k = k
    return best


def generate_parent_indices(
    *,
    n_max: int,
    seed: int,
    manifest: list[dict] | None = None,
) -> np.ndarray:
    """One stratified draw of size n_max; smaller screens are prefixes."""
    rows = manifest if manifest is not None else load_manifest()
    n_max = min(int(n_max), len(rows))
    return stratified_sample_indices(rows, n_max, seed=seed)


def resolve_sample_indices(
    cache_tag: str,
    *,
    n_max: int = config.SCREEN_N_MAX,
    write: bool = True,
) -> np.ndarray:
    """Return indices for cache_tag, nested in a parent n_max draw when possible.

    Lookup order:
      1. Residual screen cache (apples-to-apples with n=1000 gate)
      2. Local DeepONet cache
      3. Prefix of parent n_max draw (generated once, then sliced)
    """
    n, seed = parse_cache_tag(cache_tag)
    residual_path = config.RESIDUAL_CACHE_DIR / cache_tag / "sample_indices.npy"
    loaded = _load_if_exists(residual_path)
    if loaded is not None:
        if len(loaded) < n:
            raise ValueError(
                f"{residual_path} has {len(loaded)} indices, need {n}"
            )
        idx = np.asarray(loaded[:n], dtype=int)
    else:
        local_path = config.CACHE_DIR / cache_tag / "sample_indices.npy"
        loaded = _load_if_exists(local_path)
        if loaded is not None:
            if len(loaded) < n:
                raise ValueError(
                    f"{local_path} has {len(loaded)} indices, need {n}"
                )
            idx = np.asarray(loaded[:n], dtype=int)
        else:
            parent_tag = parent_cache_tag(seed, n_max)
            parent_path = config.CACHE_DIR / parent_tag / "sample_indices.npy"
            parent = _load_if_exists(parent_path)
            if parent is None:
                residual_parent = (
                    config.RESIDUAL_CACHE_DIR / parent_tag / "sample_indices.npy"
                )
                parent = _load_if_exists(residual_parent)
            if parent is None:
                parent = generate_parent_indices(n_max=n_max, seed=seed)
            nested = _largest_nested_indices(n, seed)
            if nested is not None:
                seen = {int(i) for i in nested}
                extra = [int(i) for i in parent if int(i) not in seen]
                idx = np.asarray(
                    list(np.asarray(nested, dtype=int)) + extra, dtype=int
                )
                if len(idx) < n:
                    raise ValueError(
                        f"nested+parent has {len(idx)} indices, need {n} for {cache_tag}"
                    )
                idx = idx[:n]
            else:
                if len(parent) < n:
                    raise ValueError(
                        f"parent draw has {len(parent)} indices, need {n} for {cache_tag}"
                    )
                idx = np.asarray(parent[:n], dtype=int)
            if write:
                parent_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(parent_path, np.asarray(parent, dtype=int))

    if write:
        out_dir = config.CACHE_DIR / cache_tag
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "sample_indices.npy", idx)
    return idx


def h5_basenames_for_indices(indices: np.ndarray) -> list[str]:
    manifest = load_manifest()
    names: list[str] = []
    for i in indices:
        row = manifest[int(i)]
        stored = row.get("h5_path") or row.get("path") or f"run_{int(i)}.h5"
        names.append(Path(stored).name)
    return names


def main() -> None:
    p = argparse.ArgumentParser(description="Write stratified screen sample indices")
    p.add_argument("--cache-tag", default="n2000_seed42")
    p.add_argument("--n-max", type=int, default=config.SCREEN_N_MAX)
    p.add_argument("--print-only", action="store_true")
    p.add_argument(
        "--print-h5",
        action="store_true",
        help="Print H5 basenames from the manifest (one per line)",
    )
    p.add_argument("--no-write", action="store_true")
    args = p.parse_args()
    idx = resolve_sample_indices(
        args.cache_tag, n_max=args.n_max, write=not args.no_write
    )
    print(f"{args.cache_tag}: {len(idx)} indices", file=sys.stderr, flush=True)
    if args.print_h5:
        for name in h5_basenames_for_indices(idx):
            print(name)
    elif args.print_only:
        for i in idx:
            print(int(i))


if __name__ == "__main__":
    main()
