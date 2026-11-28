"""1D VSPDB-like soil columns painted inside 2D section contacts.

gen-ai-soil-profiles is a 1D generator. Independent columns at every x would
make the 2D field identical to 1D, so each section gets **one** layered
column stretched to the local contact depth H(x). Rock bands and the
flat/ramp/step/trough contact stay in ``section_generator``.

A library npz (``data/case7a/soilgen_library.npz``) is preferred. If it is
missing, ``layered_column`` draws a simplex-thickness, depth-increasing
profile with the same interface. ``build_soilgen_library.py`` fills the npz
from the constrained-flow checkpoint in the sibling repo.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_LIBRARY = REPO / "data" / "case7a" / "soilgen_library.npz"
SOILGEN_ROOT = Path(os.environ.get("SOILGEN_ROOT", Path.home() / "gen-ai-soil-profiles"))
SOILGEN_CKPT = (
    SOILGEN_ROOT / "outputs" / "flow_matching_simplified_constrained" / "s42_e500" / "latest.pt"
)
VS_MIN = 100.0
VS_MAX = 5000.0


def layered_column(
    h: float,
    vs_surface: float,
    vs_contact: float,
    n_layers: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Simplex thicknesses and increasing Vs. Used when the flow library is absent."""
    n = max(2, int(n_layers))
    w = rng.lognormal(mean=0.0, sigma=0.45, size=n)
    h_lay = w / w.sum() * float(h)
    bounds = np.concatenate([[0.0], np.cumsum(h_lay)])
    t = (np.arange(n) + 0.5) / n
    vs = float(vs_surface) * (float(vs_contact) / max(float(vs_surface), 1.0)) ** t
    vs = vs * np.exp(rng.normal(0.0, 0.08, size=n))
    vs = np.clip(vs, VS_MIN, VS_MAX)
    vs[-1] = max(vs[-1], float(vs_contact))
    return bounds.astype(np.float64), vs.astype(np.float64)


def paint_soil(z: np.ndarray, h_x: np.ndarray, bounds: np.ndarray, vs: np.ndarray) -> np.ndarray:
    """One column stretched to each H(x). Shape (len(z), len(h_x))."""
    edges = np.asarray(bounds, dtype=np.float64)
    col = np.asarray(vs, dtype=np.float64)
    if edges[0] != 0.0 or edges[-1] <= 0:
        raise ValueError("bounds must start at 0 and end at H > 0")
    frac = edges / edges[-1]
    z_frac = np.asarray(z, dtype=np.float64)[:, None] / np.maximum(np.asarray(h_x, dtype=np.float64)[None, :], 1e-6)
    idx = np.digitize(z_frac, frac[1:], right=True)
    return col[np.clip(idx, 0, col.size - 1)]


def _load_library(path: Path) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    with np.load(path) as pack:
        return {k: pack[k] for k in pack.files}


def column_for_row(
    row: dict,
    *,
    library: dict[str, np.ndarray] | None = None,
    library_path: Path = DEFAULT_LIBRARY,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Layer bounds and Vs for this section's soil column."""
    h = float(row["soil_thickness_m"])
    vs_s = float(row["vs_surface_mps"])
    vs_c = float(row["vs_contact_mps"])
    n_layers = int(row.get("n_soil_layers", 8))
    seed = int(row.get("soilgen_seed", abs(hash(row.get("sample_id", "sec"))) % (2**31)))
    rng = np.random.default_rng(seed) if rng is None else rng
    lib = library if library is not None else _load_library(Path(library_path))
    if lib is None:
        return layered_column(h, vs_s, vs_c, n_layers, rng)
    h_lib = np.asarray(lib["h"], dtype=np.float64)
    vs_lib = np.asarray(lib["vs_z"], dtype=np.float64)
    d = ((h_lib - h) / 60.0) ** 2 + ((vs_lib - vs_s) / 200.0) ** 2
    k = int(np.argmin(d))
    n = int(lib["n"][k])
    return lib["bounds"][k, : n + 1].astype(np.float64), lib["vs"][k, :n].astype(np.float64)
