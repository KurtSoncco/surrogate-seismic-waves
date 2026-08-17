"""IID mix n-ladder indices that keep the nested n1000 seed-42 test slice clean.

Naive ``make_splits(n2000)`` train leaks ~107 of the 150 n1000 test samples
because n1000 ⊂ n2000 as *sets* but not as a cache prefix, and the two
permutations differ. Extra IID train rows are taken from n2000/n3000 samples
that are not in the n1000 corpus at all.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import config
from data import make_splits
from domain_splits import load_split
from ood_signed_cache import cache_dir_for

IID_N1000 = 1000
IID_TRAIN_M700 = 700
MIX_TAGS = {
    "M700": 700,
    "M1400": 1400,
    "M2100": 2100,
    "M7680": None,  # rest-of-IID outside n1000 + n1000 train
}
N7680_TAG = "n7680_seed42"


def extra_local_indices(
    parent_global: np.ndarray,
    child_global: np.ndarray,
    n_extra: int,
    *,
    seed: int = config.SEED,
) -> np.ndarray:
    """Local indices into ``child_global`` for samples not in ``parent_global``."""
    parent = {int(x) for x in np.asarray(parent_global).ravel()}
    extra_local = [
        i for i, g in enumerate(np.asarray(child_global).ravel()) if int(g) not in parent
    ]
    if n_extra > len(extra_local):
        raise ValueError(
            f"need {n_extra} extras but only {len(extra_local)} samples are outside parent"
        )
    rng = np.random.default_rng(seed)
    pick = rng.permutation(len(extra_local))[: int(n_extra)]
    return np.asarray([extra_local[int(i)] for i in pick], dtype=int)


def iid_n1000_split(seed: int = config.SEED) -> dict[str, np.ndarray]:
    path = config.CACHE_DIR / "splits" / f"iid_n{IID_N1000}_seed{seed}.npz"
    if path.is_file():
        return load_split(path)
    splits = make_splits(IID_N1000, seed=seed)
    return {"train": splits.train, "val": splits.val, "test": splits.test}


def mix_train_parts(
    mix_tag: str,
    *,
    seed: int = config.SEED,
    n1000_cache: Path | None = None,
    n2000_cache: Path | None = None,
    n3000_cache: Path | None = None,
) -> list[tuple[str, Path, np.ndarray]]:
    """Train parts: nested-safe IID + dipping + three_layer train slices."""
    n_iid = MIX_TAGS[mix_tag]
    n1000_cache = n1000_cache or (config.CACHE_DIR / "n1000_seed42")
    n2000_cache = n2000_cache or (config.CACHE_DIR / "n2000_seed42")
    n3000_cache = n3000_cache or (config.CACHE_DIR / "n3000_seed42")
    iid = iid_n1000_split(seed=seed)
    parts: list[tuple[str, Path, np.ndarray]] = [
        ("iid", n1000_cache, np.asarray(iid["train"], dtype=int))
    ]
    n7680_cache = config.CACHE_DIR / N7680_TAG
    if mix_tag == "M7680":
        child = np.load(n7680_cache / "sample_indices.npy")
        parent = np.load(n1000_cache / "sample_indices.npy")
        n_extra = int(len(set(int(g) for g in child) - {int(g) for g in parent}))
        extra = extra_local_indices(parent, child, n_extra, seed=seed)
        parts.append(("iid_extra", n7680_cache, extra))
    elif n_iid is not None:
        n_extra = int(n_iid) - IID_TRAIN_M700
        if n_extra > 0:
            parent = np.load(n1000_cache / "sample_indices.npy")
            if n_iid <= 1400:
                child = np.load(n2000_cache / "sample_indices.npy")
                extra = extra_local_indices(parent, child, n_extra, seed=seed)
                parts.append(("iid_extra", n2000_cache, extra))
            else:
                child = np.load(n3000_cache / "sample_indices.npy")
                extra = extra_local_indices(parent, child, n_extra, seed=seed)
                parts.append(("iid_extra", n3000_cache, extra))
    dip = load_split(config.CACHE_DIR / "splits" / f"ood_dipping_seed{seed}.npz")
    tl = load_split(config.CACHE_DIR / "splits" / f"ood_three_layer_seed{seed}.npz")
    parts.append(("ood_dipping", cache_dir_for("ood_dipping"), np.asarray(dip["train"])))
    parts.append(("ood_three_layer", cache_dir_for("ood_three_layer"), np.asarray(tl["train"])))
    return parts


def mix_val_parts(*, seed: int = config.SEED) -> list[tuple[str, Path, np.ndarray]]:
    iid = iid_n1000_split(seed=seed)
    dip = load_split(config.CACHE_DIR / "splits" / f"ood_dipping_seed{seed}.npz")
    tl = load_split(config.CACHE_DIR / "splits" / f"ood_three_layer_seed{seed}.npz")
    return [
        ("iid", config.CACHE_DIR / "n1000_seed42", np.asarray(iid["val"])),
        ("ood_dipping", cache_dir_for("ood_dipping"), np.asarray(dip["val"])),
        ("ood_three_layer", cache_dir_for("ood_three_layer"), np.asarray(tl["val"])),
    ]


def mix_test_parts(*, seed: int = config.SEED) -> dict[str, tuple[Path, np.ndarray]]:
    iid = iid_n1000_split(seed=seed)
    dip = load_split(config.CACHE_DIR / "splits" / f"ood_dipping_seed{seed}.npz")
    tl = load_split(config.CACHE_DIR / "splits" / f"ood_three_layer_seed{seed}.npz")
    return {
        "iid": (config.CACHE_DIR / "n1000_seed42", np.asarray(iid["test"])),
        "ood_dipping": (cache_dir_for("ood_dipping"), np.asarray(dip["test"])),
        "ood_three_layer": (
            cache_dir_for("ood_three_layer"),
            np.asarray(tl["test"]),
        ),
    }
