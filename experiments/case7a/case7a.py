"""Treasure Island Case 7a grid, archive cache, and LOGLO input."""

from __future__ import annotations

import io
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# 800 columns by 30 rows of 5 m quads. x = -2000 m at the west node.
N_ROW = 30
N_COL = 800
N_NODE = N_COL + 1
ELEM_M = 5.0
X_LEFT_M = -2000.0
X_RIGHT_M = 2000.0

# Operator window: element centers in [-250, 250) m, painted onto 1 m cells.
STRIP_HALF_M = 250.0
NZ_MAX = 128
NX = 500
DX_M = 1.0
DZ_M = 1.0
LX_M = 500.0
LZ_M = float(NZ_MAX)
REPEAT = int(ELEM_M / DX_M)

# Training recorders: center of the 500 m strip, ±10 stations at 15 m.
N_STATIONS = 21
NODES_EACH_SIDE = 10
STATION_SPACING_M = 15.0

# Rayleigh target ratio from the instruction sheet (0.75%).
ZETA = 0.0075
VS_NORM_EPS = 1e-6
ZETA_NORM_EPS = 1e-12

# Full section: 150 m over 128 bins, 15 m stations inside the extended mesh.
DEPTH_M = float(N_ROW * ELEM_M)
DZ_FULL_M = DEPTH_M / NZ_MAX
N_FINE_Z = int(DEPTH_M)
NX_1M = N_COL * REPEAT
ROCK_VS = 760.0
# Training zeta is ~4% in soil and ~0.4% in rock; max-normalization maps that to 1 and 0.1.
ROCK_ZETA_NORM = 0.1
GEOLOGIC_X_MIN = -650.0
GEOLOGIC_X_MAX = 1050.0
# ±1750 m are the geometric limits. Multiples of 15 m inside those limits are ±1740 m.
WINDOW_LIMIT_M = 1750.0
# Half-open x ranges (m) used to report section scores by geology.
ZONES = {
    "flat_west": (-np.inf, -650.0),
    "geologic": (-650.0, 1050.0),
    "ramp_1250": (1050.0, 1300.0),
    "flat_east": (1300.0, np.inf),
}


def zone_of(x_m: np.ndarray) -> np.ndarray:
    """Zone name for each x."""
    x = np.asarray(x_m, dtype=np.float64)
    out = np.empty(x.shape, dtype=object)
    for name, (lo, hi) in ZONES.items():
        out[(x >= lo) & (x < hi)] = name
    return out

DT_RESULTS_S = 0.001
N_TIME = 15001

REPO = Path(__file__).resolve().parents[2]
_UPLOADS = Path(
    "/home/kurt-/.cursor/projects/home-kurt-surrogate-seismic-waves/uploads"
)
DEFAULT_INSTRUCTIONS_ZIP = (
    _UPLOADS
    / "3bec909a-e3f7-4d19-a0ff-667dc6517d72"
    / "0_Model Instructions.zip"
)
DEFAULT_RESULTS_ZIP = (
    _UPLOADS / "bda9b058-fe06-4497-bcac-4a6b2a0a1980" / "Results.zip"
)
DEFAULT_SVLAB_UPLOAD = (
    _UPLOADS / "1a79cf04-cdcb-40a0-8676-ab4dc0ebe113" / "Results"
)
DEFAULT_SVLAB_DIR = REPO / "data" / "case7a" / "svlab"
DEFAULT_CACHE = REPO / "data" / "case7a" / "case7a_svlab.npz"
SVLAB_DURATION_S = 15.0
SVLAB_WEST_NODE = 24031
SVLAB_CENTER_NODE = 24431
SVLAB_FILES = (
    "Surface.0.out",
    "Acceleration_Array_Surface.0.out",
    "Acceleration_Array_Downhole.0.out",
)

_VS_NAME = "0_Model Instructions/Case7-Vs.txt"
_POISSON_NAME = "0_Model Instructions/Case7-Poisson.txt"
_RICKER_NAME = "0_Model Instructions/ricker.in"
_SURF_H_NAME = "Results/Surface_Horizontal.txt"
_SURF_V_NAME = "Results/Surface_Vertical.txt"
_DOWNHOLE_NAME = "Results/Downhole.txt"


def element_center_x(index: np.ndarray) -> np.ndarray:
    """Center x (m) of element columns. Column 0 is the westmost element."""
    j = np.asarray(index, dtype=np.float64)
    return X_LEFT_M + ELEM_M * (j + 0.5)


def crop_column_range() -> tuple[int, int]:
    """Half-open column range whose centers lie in [-250, 250) m."""
    centers = element_center_x(np.arange(N_COL))
    inside = (centers >= -STRIP_HALF_M) & (centers < STRIP_HALF_M)
    idx = np.flatnonzero(inside)
    if idx.size == 0:
        raise RuntimeError("Center strip contains no elements.")
    return int(idx[0]), int(idx[-1]) + 1


def strip_x_left_m() -> float:
    """West edge (m) of the first cropped element."""
    j0, _ = crop_column_range()
    return X_LEFT_M + ELEM_M * j0


def recorder_column_indices() -> np.ndarray:
    """1 m column indices on the 500-wide strip (training recorder layout)."""
    center = NX // 2
    step = int(round(STATION_SPACING_M / DX_M))
    lo = center - NODES_EACH_SIDE * step
    hi = center + NODES_EACH_SIDE * step
    return np.arange(lo, hi + 1, step, dtype=np.int64)


def station_x_m() -> np.ndarray:
    """Station x (m). Column index 250 on the strip is x = 0."""
    return strip_x_left_m() + recorder_column_indices().astype(np.float64)


def station_node_indices() -> np.ndarray:
    """Surface-node indices (0..800) for the 21 stations."""
    x = station_x_m()
    k_float = (x - X_LEFT_M) / ELEM_M
    k = np.rint(k_float).astype(np.int64)
    if not np.allclose(k_float, k):
        raise ValueError("Stations do not land on the 5 m nodes.")
    if np.any(k < 0) or np.any(k >= N_NODE):
        raise ValueError(f"Station nodes out of range: {k}")
    return k


def center_station_index() -> int:
    nodes = station_node_indices()
    hits = np.flatnonzero(nodes == N_COL // 2)
    if hits.size != 1:
        raise RuntimeError("Center node is not a station.")
    return int(hits[0])


def resample_block(block: np.ndarray) -> np.ndarray:
    """Paint 5 m cells onto 1 m cells. ``block`` is (n_row, n_col), surface at row 0."""
    up = np.repeat(np.repeat(block, REPEAT, axis=0), REPEAT, axis=1)
    return np.ascontiguousarray(up[:NZ_MAX])


def build_model_input(vs: np.ndarray, zeta: float = ZETA) -> np.ndarray:
    """Case 7a Vs (30, 800) -> LOGLO input (4, 128, 500).

    Channels are surface-normalized Vs, max-normalized zeta, and x/z coordinates
    normalized by the 500 m by 128 m window. Same formulas as
    ``_normalize_vs_by_surface``, ``_normalize_zeta_by_max``, and
    ``_build_sample_coord_grids`` in the GIFNO loader.
    """
    field = np.asarray(vs, dtype=np.float64)
    if field.shape != (N_ROW, N_COL):
        raise ValueError(f"Vs must be {(N_ROW, N_COL)}, got {field.shape}")
    j0, j1 = crop_column_range()
    vs_1m = resample_block(field[:, j0:j1])
    if vs_1m.shape != (NZ_MAX, NX):
        raise ValueError(f"Resampled Vs is {vs_1m.shape}, expected {(NZ_MAX, NX)}")

    surface = np.maximum(vs_1m[0:1, :], VS_NORM_EPS)
    vs_n = (vs_1m / surface).astype(np.float32)

    zeta_1m = np.full((NZ_MAX, NX), float(zeta), dtype=np.float64)
    zeta_max = float(np.max(zeta_1m))
    if zeta_max < ZETA_NORM_EPS:
        zeta_n = zeta_1m.astype(np.float32)
    else:
        zeta_n = (zeta_1m / zeta_max).astype(np.float32)

    x_cols = (np.arange(NX, dtype=np.float32) * DX_M) / LX_M
    z_rows = (np.arange(NZ_MAX, dtype=np.float32) * DZ_M) / LZ_M
    x_coord = np.broadcast_to(x_cols, (NZ_MAX, NX)).copy()
    z_coord = np.broadcast_to(z_rows[:, None], (NZ_MAX, NX)).copy()
    return np.stack([vs_n, zeta_n, x_coord, z_coord], axis=0).astype(np.float32)


def node_x_m() -> np.ndarray:
    return X_LEFT_M + ELEM_M * np.arange(N_NODE, dtype=np.float64)


def depth_bin_weights() -> np.ndarray:
    """Overlap of each 1 m cell with the 128 equal bins over 150 m. Shape (128, 150)."""
    edges = np.linspace(0.0, DEPTH_M, NZ_MAX + 1)
    fine = np.arange(N_FINE_Z + 1, dtype=np.float64)
    weights = np.zeros((NZ_MAX, N_FINE_Z), dtype=np.float64)
    for i in range(N_FINE_Z):
        left = np.maximum(edges[:-1], fine[i])
        right = np.minimum(edges[1:], fine[i + 1])
        weights[:, i] = np.maximum(0.0, right - left)
    return weights


def resample_vs_full_depth(vs: np.ndarray) -> np.ndarray:
    """(30, 800) element Vs -> (128, 4000) thickness-averaged 1 m columns."""
    field = np.asarray(vs, dtype=np.float64)
    if field.shape != (N_ROW, N_COL):
        raise ValueError(f"Vs must be {(N_ROW, N_COL)}, got {field.shape}")
    up = np.repeat(np.repeat(field, REPEAT, axis=0), REPEAT, axis=1)
    weights = depth_bin_weights()
    wsum = weights.sum(axis=1, keepdims=True)
    return (weights @ up) / wsum


def window_center_x() -> np.ndarray:
    """15 m stations whose 500 m window lies inside the extended mesh."""
    k_lo = int(np.ceil(-WINDOW_LIMIT_M / STATION_SPACING_M))
    k_hi = int(np.floor(WINDOW_LIMIT_M / STATION_SPACING_M))
    return STATION_SPACING_M * np.arange(k_lo, k_hi + 1, dtype=np.float64)


def window_column_start(x_center: float) -> int:
    """First 1 m column of the 500 m window centered at ``x_center``."""
    start = (float(x_center) - STRIP_HALF_M) - X_LEFT_M
    start_i = int(round(start))
    if abs(start - start_i) > 1e-6:
        raise ValueError(f"Window at x={x_center} does not start on a 1 m column.")
    if start_i < 0 or start_i + NX > NX_1M:
        raise ValueError(f"Window at x={x_center} leaves the mesh (start {start_i}).")
    return start_i


def window_node_index(x_center: float) -> int:
    """Surface node under the window center."""
    k = (float(x_center) - X_LEFT_M) / ELEM_M
    k_i = int(round(k))
    if abs(k - k_i) > 1e-6 or k_i < 0 or k_i >= N_NODE:
        raise ValueError(f"Window center x={x_center} is not a surface node.")
    return k_i


def build_window_input(
    vs_binned: np.ndarray,
    x_center: float,
    zeta_mode: str = "flat",
) -> np.ndarray:
    """One 500 m window of the full-depth section -> (4, 128, 500)."""
    if vs_binned.shape != (NZ_MAX, NX_1M):
        raise ValueError(f"Binned Vs must be {(NZ_MAX, NX_1M)}, got {vs_binned.shape}")
    c0 = window_column_start(x_center)
    vs_w = np.asarray(vs_binned[:, c0 : c0 + NX], dtype=np.float64)
    surface = np.maximum(vs_w[0:1, :], VS_NORM_EPS)
    vs_n = (vs_w / surface).astype(np.float32)
    if zeta_mode == "flat":
        zeta_n = np.ones((NZ_MAX, NX), dtype=np.float32)
    elif zeta_mode == "soil_rock":
        zeta_n = np.ones((NZ_MAX, NX), dtype=np.float32)
        zeta_n[vs_w > ROCK_VS] = np.float32(ROCK_ZETA_NORM)
    else:
        raise ValueError(f"Unknown zeta_mode {zeta_mode!r}")
    x_cols = (np.arange(NX, dtype=np.float32) * DX_M) / LX_M
    z_rows = (np.arange(NZ_MAX, dtype=np.float32) * DZ_FULL_M) / DEPTH_M
    x_coord = np.broadcast_to(x_cols, (NZ_MAX, NX)).copy()
    z_coord = np.broadcast_to(z_rows[:, None], (NZ_MAX, NX)).copy()
    return np.stack([vs_n, zeta_n, x_coord, z_coord], axis=0).astype(np.float32)


def paint_section_1m(vs: np.ndarray) -> np.ndarray:
    """(30, 800) element Vs -> (150, 4000) on 1 m cells, no depth resampling."""
    field = np.asarray(vs, dtype=np.float64)
    if field.shape != (N_ROW, N_COL):
        raise ValueError(f"Vs must be {(N_ROW, N_COL)}, got {field.shape}")
    return np.repeat(np.repeat(field, REPEAT, axis=0), REPEAT, axis=1)


def build_window_input_absolute(
    vs_1m: np.ndarray,
    x_center: float,
    nz_max: int = 160,
    zeta: float | np.ndarray = ZETA,
    n_prior: int = 0,
    prior_mode: str = "constant",
    prior_log: np.ndarray | None = None,
) -> np.ndarray:
    """One 500 m window in the absolute encoding (``GIFNO/absolute_inputs.py``).

    ``zeta`` is a scalar applied to the whole window, or a field with the same
    shape as ``vs_1m``. ``prior_mode='rayleigh'`` builds the 1D prior channels
    with the training Rayleigh propagator instead of frequency-independent zeta.
    """
    import sys

    gifno = str(REPO / "experiments" / "GIFNO")
    if gifno not in sys.path:
        sys.path.append(gifno)
    from absolute_inputs import encode_absolute

    if vs_1m.shape != (N_FINE_Z, NX_1M):
        raise ValueError(f"1 m Vs must be {(N_FINE_Z, NX_1M)}, got {vs_1m.shape}")
    if nz_max < N_FINE_Z:
        raise ValueError(f"nz_max={nz_max} cannot hold the {N_FINE_Z} m Case 7 column")
    c0 = window_column_start(x_center)
    vs_w = vs_1m[:, c0 : c0 + NX]
    if np.isscalar(zeta):
        zeta_w = np.full_like(vs_w, float(zeta))
    else:
        zeta_w = np.asarray(zeta, dtype=np.float64)[:, c0 : c0 + NX]
        if zeta_w.shape != vs_w.shape:
            raise ValueError(f"zeta window {zeta_w.shape} != Vs window {vs_w.shape}")
    return encode_absolute(
        vs_w,
        zeta_w,
        N_FINE_Z,
        nz_max,
        dz=DZ_M,
        dx=DX_M,
        lx=LX_M,
        n_prior=n_prior,
        prior_mode=prior_mode,
        prior_log=None if prior_log is None else np.asarray(prior_log)[:, c0 : c0 + NX],
    )


def _read_numeric(zip_path: Path, name: str) -> np.ndarray:
    with zipfile.ZipFile(zip_path) as zf, zf.open(name) as raw:
        if name.endswith(".txt") and "Results/" in name:
            frame = pd.read_csv(raw, sep="\t", header=None, dtype=np.float64)
            return frame.to_numpy(dtype=np.float64)
        text = io.TextIOWrapper(raw, encoding="utf-8")
        return np.loadtxt(text, dtype=np.float64)


def _split_time(table: np.ndarray, what: str) -> tuple[np.ndarray, np.ndarray]:
    if table.ndim != 2 or table.shape[0] != N_TIME or table.shape[1] < 2:
        raise ValueError(f"{what} has shape {table.shape}, expected ({N_TIME}, >=2)")
    time = table[:, 0]
    dt = float(np.median(np.diff(time)))
    if not np.isclose(dt, DT_RESULTS_S, rtol=0, atol=1e-6):
        raise ValueError(f"{what} dt is {dt}, expected {DT_RESULTS_S}")
    return time.astype(np.float32), table[:, 1:].astype(np.float32)


def load_archives(instructions_zip: Path, results_zip: Path) -> dict[str, np.ndarray]:
    """Read the instruction and results zips into arrays."""
    vs = _read_numeric(instructions_zip, _VS_NAME)
    poisson = _read_numeric(instructions_zip, _POISSON_NAME)
    ricker_raw = _read_numeric(instructions_zip, _RICKER_NAME)
    if vs.shape != (N_ROW, N_COL):
        raise ValueError(f"Vs shape {vs.shape} != {(N_ROW, N_COL)}")
    if poisson.shape != (N_ROW, N_COL):
        raise ValueError(f"Poisson shape {poisson.shape} != {(N_ROW, N_COL)}")
    n_ricker = int(ricker_raw.ravel()[0])
    ricker = ricker_raw.ravel()[1:]
    if ricker.size != n_ricker:
        raise ValueError(f"Ricker count {n_ricker} != {ricker.size} samples")

    time_h, surface_h = _split_time(
        _read_numeric(results_zip, _SURF_H_NAME), "Surface_Horizontal"
    )
    time_v, surface_v = _split_time(
        _read_numeric(results_zip, _SURF_V_NAME), "Surface_Vertical"
    )
    time_d, downhole_2d = _split_time(
        _read_numeric(results_zip, _DOWNHOLE_NAME), "Downhole"
    )
    if surface_h.shape[1] != N_NODE or surface_v.shape[1] != N_NODE:
        raise ValueError(
            f"Surface channels {surface_h.shape[1]}, {surface_v.shape[1]} != {N_NODE}"
        )
    if downhole_2d.shape[1] != 1:
        raise ValueError(f"Downhole channels {downhole_2d.shape[1]} != 1")
    if not (np.allclose(time_h, time_v) and np.allclose(time_h, time_d)):
        raise ValueError("Surface and downhole time axes differ.")

    return {
        "vs": vs.astype(np.float32),
        "poisson": poisson.astype(np.float32),
        "ricker": ricker.astype(np.float32),
        "time": time_h,
        "surface_h": surface_h,
        "surface_v": surface_v,
        "downhole": downhole_2d[:, 0].astype(np.float32),
        "dt": np.float32(DT_RESULTS_S),
    }


def load_svlab_motion(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Real-ESSI ``.out`` → node ids and motion ``(n_time, n_node, n_dof)``.

    Header is ``n_node n_col n_eq n_step``. The body row count is the time
    length; it can be one less than ``n_step``.
    """
    path = Path(path)
    with path.open() as raw:
        header = raw.readline().split()
        if len(header) < 4:
            raise ValueError(f"{path} header is {header!r}")
        n_node, n_col, _, n_step = (int(v) for v in header[:4])
        nodes = np.empty(n_node, dtype=np.int64)
        for i in range(n_node):
            parts = raw.readline().split()
            if not parts:
                raise ValueError(f"{path} ended in the node table at row {i}")
            nodes[i] = int(parts[0])
        body = np.loadtxt(raw, dtype=np.float64)
    if body.size == 0:
        raise ValueError(f"{path} has no time samples")
    if body.ndim == 1:
        body = body.reshape(1, -1)
    if body.shape[1] != n_col:
        raise ValueError(f"{path} has {body.shape[1]} columns, expected {n_col}")
    n_dof = n_col // n_node
    if n_dof < 1 or n_node * n_dof != n_col:
        raise ValueError(f"{path} n_col={n_col} is not n_node={n_node} times n_dof")
    n_time = int(body.shape[0])
    if n_time not in (n_step, n_step - 1):
        raise ValueError(f"{path} has {n_time} samples, header n_step={n_step}")
    return nodes, body.reshape(n_time, n_node, n_dof)


def install_svlab(
    src: Path = DEFAULT_SVLAB_UPLOAD,
    dest: Path = DEFAULT_SVLAB_DIR,
) -> Path:
    """Copy the three SVLAB ``.out`` files into ``data/case7a/svlab/``."""
    src = Path(src)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    missing = [name for name in SVLAB_FILES if not (src / name).is_file()]
    if missing:
        raise FileNotFoundError(f"SVLAB source {src} missing {missing}")
    for name in SVLAB_FILES:
        shutil.copy2(src / name, dest / name)
    return dest


def _instruction_fields(instructions_zip: Path) -> dict[str, np.ndarray]:
    vs = _read_numeric(instructions_zip, _VS_NAME)
    poisson = _read_numeric(instructions_zip, _POISSON_NAME)
    ricker_raw = _read_numeric(instructions_zip, _RICKER_NAME)
    if vs.shape != (N_ROW, N_COL):
        raise ValueError(f"Vs shape {vs.shape} != {(N_ROW, N_COL)}")
    if poisson.shape != (N_ROW, N_COL):
        raise ValueError(f"Poisson shape {poisson.shape} != {(N_ROW, N_COL)}")
    n_ricker = int(ricker_raw.ravel()[0])
    ricker = ricker_raw.ravel()[1:]
    if ricker.size != n_ricker:
        raise ValueError(f"Ricker count {n_ricker} != {ricker.size} samples")
    return {
        "vs": vs.astype(np.float32),
        "poisson": poisson.astype(np.float32),
        "ricker": ricker.astype(np.float32),
    }


def load_svlab(
    svlab_dir: Path = DEFAULT_SVLAB_DIR,
    instructions_zip: Path = DEFAULT_INSTRUCTIONS_ZIP,
) -> dict[str, np.ndarray]:
    """SVLAB surface/downhole plus instruction-sheet Vs and Ricker."""
    svlab_dir = Path(svlab_dir)
    pack = _instruction_fields(instructions_zip)
    nodes, surf = load_svlab_motion(svlab_dir / "Surface.0.out")
    if surf.shape[1] != N_NODE:
        raise ValueError(f"SVLAB surface nodes {surf.shape[1]} != {N_NODE}")
    if int(nodes[0]) != SVLAB_WEST_NODE:
        raise ValueError(f"west node is {int(nodes[0])}, expected {SVLAB_WEST_NODE}")
    hits = np.flatnonzero(nodes == SVLAB_CENTER_NODE)
    if hits.size != 1:
        raise ValueError(f"center node {SVLAB_CENTER_NODE} hits {hits.size} times")
    center = int(hits[0])
    if center != N_COL // 2:
        raise ValueError(f"center node index {center} != {N_COL // 2}")

    arr_nodes, arr_s = load_svlab_motion(svlab_dir / "Acceleration_Array_Surface.0.out")
    if int(arr_nodes[0]) != SVLAB_CENTER_NODE:
        raise ValueError(f"array surface node {int(arr_nodes[0])} != {SVLAB_CENTER_NODE}")
    n_match = min(surf.shape[0], arr_s.shape[0])
    if not np.allclose(surf[:n_match, center, 0], arr_s[:n_match, 0, 0]):
        raise ValueError("Acceleration_Array_Surface.0.out does not match Surface.0.out at x=0")

    _, arr_d = load_svlab_motion(svlab_dir / "Acceleration_Array_Downhole.0.out")
    n_time = int(surf.shape[0])
    if arr_d.shape[0] != n_time:
        raise ValueError(f"downhole length {arr_d.shape[0]} != surface {n_time}")
    dt = SVLAB_DURATION_S / float(n_time - 1)
    time = (np.arange(n_time, dtype=np.float64) * dt).astype(np.float32)
    return {
        **pack,
        "time": time,
        "surface_h": surf[:, :, 0].astype(np.float32),
        "surface_v": surf[:, :, 1].astype(np.float32),
        "downhole": arr_d[:, 0, 0].astype(np.float32),
        "dt": np.float32(dt),
        "svlab_nodes": nodes.astype(np.int32),
        "source": np.array("svlab"),
    }


def ensure_cache(
    cache_path: Path = DEFAULT_CACHE,
    instructions_zip: Path = DEFAULT_INSTRUCTIONS_ZIP,
    results_zip: Path = DEFAULT_RESULTS_ZIP,
    *,
    svlab_dir: Path = DEFAULT_SVLAB_DIR,
    force: bool = False,
) -> dict[str, np.ndarray]:
    """Load the SVLAB Case 7 cache, building it if needed. OpenSees zip is not used."""
    del results_zip
    cache_path = Path(cache_path)
    svlab_dir = Path(svlab_dir)
    if cache_path.is_file() and not force:
        with np.load(cache_path) as cached:
            pack = {key: cached[key] for key in cached.files}
        source = str(np.asarray(pack.get("source", "")).astype(str).reshape(-1)[0]) if "source" in pack else ""
        if source == "svlab":
            return pack
    surface = svlab_dir / "Surface.0.out"
    if not surface.is_file():
        install_svlab(dest=svlab_dir)
    if not surface.is_file():
        raise FileNotFoundError(
            f"SVLAB ground truth missing at {svlab_dir}. Copy Surface.0.out there."
        )
    pack = load_svlab(svlab_dir, instructions_zip)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **pack)
    return pack
