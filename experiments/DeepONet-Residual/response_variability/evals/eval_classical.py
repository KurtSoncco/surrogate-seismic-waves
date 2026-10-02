#!/usr/bin/env python3
"""Score Toro / Passeri / Dmult on nested packs (IID or OOD) from H5 + Haskell.

Pretell is reused from presentation packs when present. Writes
``results/response_variability/eval_bias/{domain}_classical.npz``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

_EXP = Path(__file__).resolve().parents[2]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

import config  # noqa: E402

from response_variability.plots.plot_presentation import (  # noqa: E402
    DOMAIN_SPECS,
    load_pack,
    pack_path,
)
from response_variability.seiskit_arms import (  # noqa: E402
    attach_dmult_p84,
    attach_pretell_p84,
    ensure_seiskit,
    hallal_dmin_geomean_tf,
    hallal_geomean_tf,
)

OUT_DIR = config.RESULTS_DIR / "response_variability" / "eval_bias"
PACK_DIR = config.RESULTS_DIR / "presentation"
SEISKIT_PRED = (
    config.RESULTS_DIR / "response_variability" / "seiskit" / "predictions.npz"
)


def add_classical_1d_arms(
    pack: dict[str, np.ndarray],
    *,
    n_hallal_seeds: int = 40,
    skip_if_present: bool = True,
) -> dict[str, np.ndarray]:
    """Add Toro / Passeri / Dmult geomeans. Requires vs1, H, cov, vs2."""
    need = ("tf_toro", "tf_passeri", "tf_dmult")
    if skip_if_present and all(k in pack for k in need):
        return attach_dmult_p84(attach_pretell_p84(pack))
    ensure_seiskit()
    freq = pack["freq"]
    n = int(np.asarray(pack["tf_opensees"]).shape[0])
    n_freq = int(freq.shape[0])
    tf_toro = np.empty((n, n_freq), dtype=np.float64)
    tf_passeri = np.empty((n, n_freq), dtype=np.float64)
    tf_dmult = np.empty((n, n_freq), dtype=np.float64)
    sig_toro = np.empty((n, n_freq), dtype=np.float64)
    sig_passeri = np.empty((n, n_freq), dtype=np.float64)
    sig_dmult = np.empty((n, n_freq), dtype=np.float64)
    for i in tqdm(range(n), desc="Toro/Passeri/Dmult", leave=False):
        vs1 = float(pack["vs1"][i])
        H = float(pack["H"][i])
        cov = float(pack["cov"][i])
        vs2 = float(pack["vs2"][i])
        xi = float(pack["xi_mean"][i]) if "xi_mean" in pack else config.DEFAULT_XI_TREND
        if not np.isfinite(xi) or xi <= 0:
            xi = config.DEFAULT_XI_TREND
        geo_t, sig_t = hallal_geomean_tf(
            freq=freq,
            vs1=vs1,
            H=H,
            cov=cov,
            vs2=vs2,
            xi=xi,
            n_seeds=n_hallal_seeds,
            kind="toro",
        )
        geo_p, sig_p = hallal_geomean_tf(
            freq=freq,
            vs1=vs1,
            H=H,
            cov=cov,
            vs2=vs2,
            xi=xi,
            n_seeds=n_hallal_seeds,
            kind="passeri",
        )
        geo_d, sig_d = hallal_dmin_geomean_tf(freq=freq, vs1=vs1, H=H, cov=cov, vs2=vs2)
        tf_toro[i], sig_toro[i] = geo_t, np.asarray(sig_t, dtype=np.float64)
        tf_passeri[i], sig_passeri[i] = geo_p, np.asarray(sig_p, dtype=np.float64)
        tf_dmult[i], sig_dmult[i] = geo_d, np.asarray(sig_d, dtype=np.float64)
    pack = dict(pack)
    pack["tf_toro"] = tf_toro
    pack["tf_passeri"] = tf_passeri
    pack["tf_dmult"] = tf_dmult
    pack["sigma_ln_toro"] = sig_toro
    pack["sigma_ln_passeri"] = sig_passeri
    pack["sigma_ln_dmult"] = sig_dmult
    pack["n_hallal_seeds"] = np.array(n_hallal_seeds)
    return attach_dmult_p84(attach_pretell_p84(pack))


CLASSICAL_TF_KEYS = (
    "tf_toro",
    "tf_passeri",
    "tf_dmult",
    "tf_dmult_p84",
    "tf_pretell",
    "tf_pretell_p84",
    "sigma_ln_toro",
    "sigma_ln_passeri",
    "sigma_ln_dmult",
    "sigma_ln_pretell",
)


def merge_classical_into_pack(
    pack: dict[str, np.ndarray],
    domain: str,
    *,
    out_dir: Path | None = None,
) -> dict[str, np.ndarray]:
    """Overlay Toro / Passeri / Dmult / Pretell from seiskit or classical NPZ.

    Does not copy per-recorder ``tf_haskell_column`` (not a Pretell arm).
    """
    n = int(np.asarray(pack["tf_opensees"]).shape[0])
    out = dict(pack)
    candidates: list[Path] = []
    if domain == "iid":
        candidates.append(SEISKIT_PRED)
    candidates.append(classical_path(out_dir or OUT_DIR, domain))
    for path in candidates:
        if not path.is_file():
            continue
        other = np.load(path, allow_pickle=True)
        for key in CLASSICAL_TF_KEYS:
            if key not in other.files:
                continue
            arr = other[key]
            if np.asarray(arr).shape[0] != n:
                continue
            out[key] = arr
    return attach_dmult_p84(attach_pretell_p84(out))


def classical_path(out_dir: Path, domain: str) -> Path:
    return Path(out_dir) / f"{DOMAIN_SPECS[domain]['pack_name']}_classical.npz"


def run_domain(
    domain: str,
    *,
    pack_dir: Path,
    out_dir: Path,
    n_hallal_seeds: int,
    skip_if_present: bool,
) -> Path:
    src = pack_path(pack_dir, domain)
    if not src.is_file():
        raise FileNotFoundError(f"Missing presentation pack {src}")
    dest = classical_path(out_dir, domain)
    if skip_if_present and dest.is_file():
        return dest
    pack = load_pack(src)
    from response_variability.covariates import attach_h5_covariates

    pack = attach_h5_covariates(pack, domain=domain)
    pack = add_classical_1d_arms(
        pack, n_hallal_seeds=n_hallal_seeds, skip_if_present=False
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **{k: pack[k] for k in pack})
    print(f"Wrote {dest}", flush=True)
    return dest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack-dir", type=Path, default=PACK_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--n-hallal-seeds", type=int, default=40)
    p.add_argument(
        "--domains",
        nargs="+",
        default=["dipping", "three_layer"],
        choices=list(DOMAIN_SPECS),
    )
    p.add_argument("--skip-if-present", action="store_true")
    args = p.parse_args()
    for domain in args.domains:
        run_domain(
            domain,
            pack_dir=args.pack_dir,
            out_dir=args.out_dir,
            n_hallal_seeds=args.n_hallal_seeds,
            skip_if_present=args.skip_if_present,
        )


if __name__ == "__main__":
    main()
