"""Windows on the metric canvas, queried at the station's real x.

Runs (IID, dipping, three-layer) rebuild the 1500 m mesh from the stored
500 m strip. Manifold sections are already the interior: the pads were dropped
when the labels were written. Training sections are supervised every 20 m.
The column 1D prior travels with the label and is added outside the network.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _loader_threads(_worker_id: int) -> None:
    """One compute thread per loader process, so sixteen workers do not each take every core."""
    torch.set_num_threads(1)

from encoding import W_MAX_M, W_MIN_M, encode, query_columns

IID_MESH_M = 1500
IID_STRIP0 = 500
SECTION_MARGIN_M = 5
STATION_MARGIN_M = 5
K_MAX = 512
TRAIN_STRIDE_M = 20
RUN_STATION_SPACING_M = 15.0
EVAL_WIDTHS = (500, 1000, 1500, 2000)
IID_EVAL_WIDTHS = (500, 1000, 1500)
KIND_RUN, KIND_SECTION, KIND_COV0 = 0, 1, 2
# Unified training is stage 3. Stages 0, 1 and 2 keep their existing pools.
STAGE_UNIFIED = 3
BLOCKED_TRAIN_SPLITS = frozenset({"test", "blind", "external", "external_blind"})
FAMILY = {"iid": 0, "ood_dipping": 1, "ood_three_layer": 2, "manifold": 3}
FAMILY_NAME = {v: k for k, v in FAMILY.items()}
# Mean lateral std of log Vs. A repeated column sits at 0; 0.01 keeps the
# smooth one-feature flats and drops a contact that actually moves.
NEAR_1D_LOG_STD = 0.01
# A trough here is a cosine bowl in the soil-rock contact. It is not a basin
# measured at a real site, so those sections are kept on disk and left out of
# training and scoring. Feature index 3 is section_generator.FEATURES.
TROUGH_FEATURE = 3
MANIFESTS = (
    Path(__file__).resolve().parent / "sections" / "manifold150.csv",
    Path(__file__).resolve().parent / "sections" / "manifold150_wide.csv",
)


def section_has_trough(feature_name: str, extra_features: str = "[]") -> bool:
    """True when the contact itself, or a second feature on it, is a cosine bowl."""
    if str(feature_name) == "trough":
        return True
    text = extra_features if isinstance(extra_features, str) else "[]"
    if text in ("", "[]", "nan"):
        return False
    try:
        extras = json.loads(text)
    except json.JSONDecodeError:
        return False
    return any(int(item.get("feature", -1)) == TROUGH_FEATURE for item in extras)


def trough_sample_ids(manifests: tuple[Path, ...] = MANIFESTS) -> set[str]:
    """Sample ids whose solved section contains a trough. Missing files contribute nothing."""
    import pandas as pd

    dropped: set[str] = set()
    for path in manifests:
        if not path.is_file():
            continue
        table = pd.read_csv(path)
        for row in table.itertuples(index=False):
            extra = getattr(row, "extra_features", "[]")
            extra = "[]" if extra is None or (isinstance(extra, float)) else str(extra)
            if section_has_trough(str(row.feature_name), extra):
                dropped.add(str(row.sample_id))
    return dropped


def lateral_log_std(vs: np.ndarray) -> float:
    """Mean over depth of the lateral standard deviation of log Vs."""
    field = np.log(np.maximum(np.asarray(vs, dtype=np.float64), 1.0))
    return float(field.std(axis=1).mean())


@dataclass
class Campaign:
    name: str
    vs: np.ndarray
    zeta: np.ndarray
    n_active: np.ndarray
    log_tf: np.ndarray
    log_tf1d: np.ndarray
    valid: np.ndarray
    split: np.ndarray
    rec_x: np.ndarray
    f1: np.ndarray
    f2: np.ndarray
    group: np.ndarray

    @property
    def n(self) -> int:
        return int(self.n_active.shape[0])


@dataclass
class Section:
    name: str
    vs: np.ndarray
    zeta: np.ndarray
    x_m: np.ndarray
    log_tf: np.ndarray
    log_tf1d: np.ndarray
    split: str
    f1_hz: float = 1.0
    f2_hz: float = 3.0
    stratum: str = ""
    group_id: str = ""
    group: int = 0
    stratum_code: int = 0
    damping: str = ""

    @property
    def lx(self) -> int:
        return int(self.vs.shape[1])


def _load_campaign(name: str, root: Path) -> Campaign:
    n = np.load(root / "n_active.npy").shape[0]
    log_tf = np.load(root / "log_tf.npy", mmap_mode="r")
    prior_path = root / "log_tf1d.npy"
    log_tf1d = np.load(prior_path, mmap_mode="r") if prior_path.is_file() else np.zeros_like(log_tf)
    f1 = np.load(root / "rayleigh_f1.npy") if (root / "rayleigh_f1.npy").is_file() else np.ones(n, np.float32)
    f2 = np.load(root / "rayleigh_f2.npy") if (root / "rayleigh_f2.npy").is_file() else np.full(n, 3.0, np.float32)
    group = np.load(root / "group.npy") if (root / "group.npy").is_file() else np.arange(n)
    return Campaign(
        name, np.load(root / "vs.npy", mmap_mode="r"), np.load(root / "zeta.npy", mmap_mode="r"),
        np.load(root / "n_active.npy"), log_tf, log_tf1d, np.load(root / "valid.npy"),
        np.load(root / "split.npy"), np.load(root / "rec_x.npy"), f1, f2, group,
    )


class Cache:
    def __init__(self, root: Path):
        root = Path(root)
        self.root = root
        self.freq = np.load(root / "freq.npy")
        self.campaigns: list[Campaign] = []
        for name in ("iid", "ood_dipping", "ood_three_layer"):
            path = root / name
            if (path / "vs.npy").is_file():
                self.campaigns.append(_load_campaign(name, path))
        self.run_of: list[tuple[int, int]] = []
        for ci, camp in enumerate(self.campaigns):
            self.run_of.extend((ci, i) for i in range(camp.n))
        self.sections: list[Section] = []
        self.dropped_troughs: list[str] = []
        strata: dict[str, int] = {}
        groups: dict[str, int] = {}
        troughs = trough_sample_ids()
        sec_dir = root / "sections"
        if sec_dir.is_dir():
            for p in sorted(sec_dir.glob("*.npz")):
                sample_id = p.stem.split("__")[-1]
                if sample_id in troughs:
                    self.dropped_troughs.append(sample_id)
                    continue
                d = np.load(p, allow_pickle=True)
                stratum = str(d["stratum"]) if "stratum" in d.files else ""
                group_id = str(d["group_id"]) if "group_id" in d.files else p.stem
                strata.setdefault(stratum, len(strata))
                groups.setdefault(group_id, len(groups))
                log_tf = d["log_tf"]
                prior = d["log_tf1d"] if "log_tf1d" in d.files else np.zeros_like(log_tf)
                self.sections.append(Section(
                    p.stem, d["vs"], d["zeta"], d["x_m"], log_tf, prior, str(d["split"]),
                    float(d["f1_hz"]) if "f1_hz" in d.files else 1.0,
                    float(d["f2_hz"]) if "f2_hz" in d.files else 3.0,
                    stratum, group_id, groups[group_id], strata[stratum],
                    str(d["damping"]) if "damping" in d.files else "",
                ))
        self.stratum_name = {v: k for k, v in strata.items()}

    def runs(self, split: str, families: tuple[str, ...] | None = None) -> np.ndarray:
        out = []
        for flat, (ci, i) in enumerate(self.run_of):
            camp = self.campaigns[ci]
            if families and camp.name not in families:
                continue
            if camp.split[i] == split and bool(camp.valid[i]):
                out.append(flat)
        return np.asarray(out, dtype=np.int64)

    def section_indices(self, split: str) -> np.ndarray:
        return np.array([i for i, s in enumerate(self.sections) if s.split == split], dtype=np.int64)

    def campaign(self, flat: int) -> tuple[Campaign, int]:
        ci, i = self.run_of[int(flat)]
        return self.campaigns[ci], i


def iid_start_range(width: int, rec_mesh: np.ndarray) -> tuple[int, int]:
    lo = max(0, int(rec_mesh.max()) + STATION_MARGIN_M + 1 - width)
    hi = min(IID_MESH_M - width, int(rec_mesh.min()) - STATION_MARGIN_M)
    return lo, hi


def _subsample(x_local: np.ndarray, stride_m: int) -> np.ndarray:
    if stride_m <= 0:
        return np.ones(x_local.shape[0], dtype=bool)
    return np.isclose(np.mod(x_local, stride_m), 0.0, atol=0.2)


def resample_columns(field: np.ndarray, src: np.ndarray) -> np.ndarray:
    """Linear sample of (Z, N) at fractional column indices. The edge column repeats."""
    n = int(field.shape[1])
    x = np.clip(np.asarray(src, dtype=np.float64), 0.0, n - 1)
    x0 = np.floor(x).astype(np.int64)
    x1 = np.minimum(x0 + 1, n - 1)
    t = (x - x0)[None, :]
    return np.asarray(field, dtype=np.float64)[:, x0] * (1.0 - t) + np.asarray(field, dtype=np.float64)[:, x1] * t


def make_cov0_window(cache: Cache, encoding: str, index: int, width: int = W_MIN_M):
    """One IID column repeated across ``width``. The label equals the 1D prior, so g is 0."""
    camp, j = cache.campaign(index)
    n_active = int(camp.n_active[j])
    col = int(np.asarray(camp.vs.shape[-1]) // 2)
    vs_col = np.asarray(camp.vs[j, :n_active, col], dtype=np.float64)
    zeta_col = np.asarray(camp.zeta[j, :n_active, col], dtype=np.float64)
    vs = np.repeat(vs_col[:, None], width, axis=1)
    zeta = np.repeat(zeta_col[:, None], width, axis=1)
    station = int(np.argmin(np.abs(np.asarray(camp.rec_x) - col)))
    curve = np.asarray(camp.log_tf1d[j, station], dtype=np.float32)
    x_local = np.arange(STATION_MARGIN_M, width - STATION_MARGIN_M + 1e-6, RUN_STATION_SPACING_M)
    log_tf = np.repeat(curve[None, :], x_local.size, axis=0)
    f1_hz, f2_hz = float(camp.f1[j]), float(camp.f2[j])
    x = encode(encoding, vs, zeta, n_active, f1_hz=f1_hz, f2_hz=f2_hz)
    k = x_local.size
    n_freq = log_tf.shape[-1]
    query = np.zeros(K_MAX, dtype=np.float32)
    tf = np.zeros((K_MAX, n_freq), dtype=np.float32)
    prior = np.zeros((K_MAX, n_freq), dtype=np.float32)
    valid = np.zeros(K_MAX, dtype=bool)
    x_abs = np.zeros(K_MAX, dtype=np.float32)
    if k:
        query[:k] = query_columns(encoding, x_local, width).astype(np.float32)
        tf[:k] = log_tf
        prior[:k] = log_tf
        valid[:k] = True
        x_abs[:k] = x_local.astype(np.float32)
    info = np.array([KIND_COV0, index, width, 0, FAMILY[camp.name], int(camp.group[j]), 0, 0], dtype=np.int64)
    return (
        torch.from_numpy(np.ascontiguousarray(x)),
        torch.from_numpy(query),
        torch.from_numpy(tf),
        torch.from_numpy(prior),
        torch.from_numpy(valid),
        torch.from_numpy(info),
        torch.from_numpy(x_abs),
    )


def apply_mirror(
    vs: np.ndarray, zeta: np.ndarray, x_local: np.ndarray, width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact left-right flip used by training.

    Columns of Vs and zeta reverse. A station at ``x`` is queried at
    ``width - x``. The transfer-function row stays with that station; this
    function does not reorder it.
    """

    return (
        np.ascontiguousarray(np.asarray(vs)[:, ::-1]),
        np.ascontiguousarray(np.asarray(zeta)[:, ::-1]),
        float(width) - np.asarray(x_local, dtype=np.float64),
    )


def make_window(
    cache: Cache, encoding: str, kind: int, index: int, width: int, start: int,
    stride_m: int, mirror: bool, offset_m: float = 0.0,
):
    offset_m = float(offset_m)
    flags = 0
    if kind == KIND_RUN:
        camp, j = cache.campaign(index)
        n_active = int(camp.n_active[j])
        src = np.arange(width, dtype=np.float64) + (start + offset_m - IID_STRIP0)
        vs = resample_columns(np.asarray(camp.vs[j, :n_active], dtype=np.float64), src)
        zeta = resample_columns(np.asarray(camp.zeta[j, :n_active], dtype=np.float64), src)
        f1_hz, f2_hz = float(camp.f1[j]), float(camp.f2[j])
        x_local = (IID_STRIP0 + camp.rec_x - start - offset_m).astype(np.float64)
        log_tf = np.asarray(camp.log_tf[j], dtype=np.float32)
        log_prior = np.asarray(camp.log_tf1d[j], dtype=np.float32)
        family = FAMILY[camp.name]
        group = int(camp.group[j])
        stratum = 0
        keep = (x_local >= STATION_MARGIN_M) & (x_local <= width - STATION_MARGIN_M)
    else:
        sec = cache.sections[index]
        n_active = sec.vs.shape[0]
        width = min(width, sec.lx)
        start = min(start, max(0, sec.lx - width))
        src = np.arange(width, dtype=np.float64) + (start + offset_m)
        vs = resample_columns(np.asarray(sec.vs, dtype=np.float64), src)
        zeta = resample_columns(np.asarray(sec.zeta, dtype=np.float64), src)
        x_local = sec.x_m.astype(np.float64) - start - offset_m
        keep = (x_local >= STATION_MARGIN_M) & (x_local <= width - STATION_MARGIN_M)
        keep &= _subsample(sec.x_m.astype(np.float64), stride_m)
        f1_hz, f2_hz = sec.f1_hz, sec.f2_hz
        log_tf = sec.log_tf
        log_prior = sec.log_tf1d
        family = FAMILY["manifold"]
        group = int(sec.group)
        stratum = int(sec.stratum_code)
        if lateral_log_std(sec.vs) < NEAR_1D_LOG_STD:
            flags |= 1
        if sec.split == "blind":
            flags |= 2
    idx = np.flatnonzero(keep)
    if idx.size > K_MAX:
        idx = idx[np.linspace(0, idx.size - 1, K_MAX).round().astype(int)]
    x_local = x_local[idx]
    log_tf = np.asarray(log_tf[idx], dtype=np.float32)
    log_prior = np.asarray(log_prior[idx], dtype=np.float32)
    if mirror:
        vs, zeta, x_local = apply_mirror(vs, zeta, x_local, width)
    x = encode(encoding, vs, zeta, n_active, f1_hz=f1_hz, f2_hz=f2_hz)
    k = x_local.size
    query = np.zeros(K_MAX, dtype=np.float32)
    tf = np.zeros((K_MAX, log_tf.shape[-1]), dtype=np.float32)
    prior = np.zeros((K_MAX, log_tf.shape[-1]), dtype=np.float32)
    valid = np.zeros(K_MAX, dtype=bool)
    x_abs = np.zeros(K_MAX, dtype=np.float32)
    if k:
        query[:k] = query_columns(encoding, x_local, width).astype(np.float32)
        tf[:k] = log_tf
        prior[:k] = log_prior
        valid[:k] = True
        shift_back = 0.0 if mirror else offset_m
        x_abs[:k] = (x_local + start + shift_back).astype(np.float32)
    info = np.array([kind, index, width, start, family, group, flags, stratum], dtype=np.int64)
    return (
        torch.from_numpy(np.ascontiguousarray(x)),
        torch.from_numpy(query),
        torch.from_numpy(tf),
        torch.from_numpy(prior),
        torch.from_numpy(valid),
        torch.from_numpy(info),
        torch.from_numpy(x_abs),
    )


class WindowSet(Dataset):
    """One record per row. Width and start are drawn in ``__getitem__`` unless ``fixed``."""

    def __init__(self, cache: Cache, encoding: str, records: list[dict], stage: int, seed: int, mirror: bool, jitter: bool = False):
        self.cache = cache
        self.encoding = encoding
        self.records = records
        self.stage = int(stage)
        self.seed = int(seed)
        self.mirror = bool(mirror)
        self.jitter = bool(jitter)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def _draw(self, rec: dict, rng: np.random.Generator) -> tuple[int, int]:
        camp, _ = self.cache.campaign(rec["index"])
        # Stage 1 jitters width on the IID pads only. OOD stays at 500 m.
        if self.stage == 1 and rec.get("family") == "iid":
            width = int(rng.integers(W_MIN_M, min(IID_MESH_M, 1500) + 1))
        else:
            width = W_MIN_M
        lo, hi = iid_start_range(width, IID_STRIP0 + np.asarray(camp.rec_x))
        if hi < lo:
            width = W_MIN_M
            lo, hi = iid_start_range(width, IID_STRIP0 + np.asarray(camp.rec_x))
        hi = max(hi, lo)
        return width, int(rng.integers(lo, hi + 1))

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        rng = np.random.default_rng([self.seed, self.epoch, int(rec["draw"])])
        if rec.get("fixed"):
            width, start = int(rec["width"]), int(rec["start"])
        else:
            width, start = self._draw(rec, rng)
        mirror = self.mirror and bool(rec.get("allow_mirror", True)) and rng.random() < 0.5
        if rec["kind"] == "cov0":
            return make_cov0_window(self.cache, self.encoding, int(rec["index"]), int(rec.get("width", W_MIN_M)))
        kind = KIND_SECTION if rec["kind"] == "section" else KIND_RUN
        offset = 0.0
        if self.jitter:
            spacing = TRAIN_STRIDE_M if rec["kind"] == "section" else RUN_STATION_SPACING_M
            offset = float(rng.uniform(0.0, spacing))
        return make_window(
            self.cache, self.encoding, kind, int(rec["index"]), width, start, int(rec.get("stride", 0)), mirror, offset,
        )


def _cov0_records(cache: Cache, split: str, limit: int = 32) -> list[dict]:
    """One repeated-column row per IID group, up to ``limit``. The residual target is zero."""
    rows = []
    seen: set[int] = set()
    for flat in cache.runs(split, ("iid",)):
        camp, j = cache.campaign(int(flat))
        group = int(camp.group[j])
        if group in seen:
            continue
        seen.add(group)
        rows.append({
            "kind": "cov0", "index": int(flat), "draw": 0, "fixed": True,
            "width": W_MIN_M, "start": 0, "stride": 0, "allow_mirror": False,
            "family": "cov0", "group": group, "pair": False, "near_1d": True,
        })
        if len(rows) >= limit:
            break
    return rows


def nested_order(ids: list, seed: int) -> list:
    """One seeded permutation. Prefixes of the result are nested subsets."""
    values = sorted(set(ids))
    if not values:
        return []
    rng = np.random.default_rng(int(seed))
    return [values[int(i)] for i in rng.permutation(len(values))]


def nested_prefix(ids: list, frac: float, seed: int, *, widths: dict | None = None) -> list:
    """Prefix of ``nested_order``. A larger fraction contains every smaller one.

    When ``widths`` is set, the prefix is extended along that same order until
    every width present in ``ids`` is represented. The extension keeps the
    nesting: the repaired 25% set is still a subset of the repaired 50% set.
    """
    order = nested_order(list(ids), seed)
    if not order:
        return []
    frac = float(frac)
    if frac <= 0 or frac > 1:
        raise ValueError(f"fraction must be in (0, 1], got {frac}")
    n_keep = len(order) if frac >= 1.0 else min(len(order), max(1, int(round(frac * len(order)))))
    chosen = list(order[:n_keep])
    if not widths:
        return chosen
    present = {int(widths[group]) for group in order if group in widths}
    have = {int(widths[group]) for group in chosen if group in widths}
    for group in order[n_keep:]:
        if have >= present:
            break
        width = int(widths[group])
        if width not in have:
            chosen.append(group)
            have.add(width)
    return chosen


def choose_iid_flats(
    by_group: dict[int, list[int]], group_frac: float, seeds_per_group: int | None, seed: int,
) -> set[int]:
    """Geology groups, then seeds inside a group. Smaller draws are prefixes of larger ones."""
    if not by_group:
        return set()
    frac = float(group_frac)
    if frac <= 0 or frac > 1:
        raise ValueError(f"iid group fraction must be in (0, 1], got {group_frac}")
    chosen = nested_prefix(list(by_group), frac, seed)
    kept: set[int] = set()
    for group in chosen:
        flats = nested_order([int(i) for i in by_group[int(group)]], int(seed) * 1_000_003 + int(group))
        if seeds_per_group is None or int(seeds_per_group) >= len(flats):
            kept.update(flats)
            continue
        n_seeds = int(seeds_per_group)
        if n_seeds < 1:
            raise ValueError(f"iid seeds per group must be positive, got {seeds_per_group}")
        kept.update(int(i) for i in flats[:n_seeds])
    return kept


def manifold_pair_group_ids(cache: Cache) -> list[str]:
    """Group ids that occur twice. Those 22 pairs are the repeated-seed sections."""
    counts: dict[str, int] = {}
    for sec in cache.sections:
        if sec.split == "blind":
            continue
        counts[sec.group_id] = counts.get(sec.group_id, 0) + 1
    return sorted(gid for gid, n in counts.items() if n >= 2)


def manifold_folds(group_ids: list[str], n_folds: int = 5, seed: int = 0) -> list[list[str]]:
    """Partition every pair group into ``n_folds`` held-out sets.

    Twenty-two groups do not divide into five sets of four. Round-robin leaves
    every group out once, and the fold sizes differ by one (5, 5, 4, 4, 4).
    """
    if int(n_folds) < 1:
        raise ValueError(f"n_folds must be positive, got {n_folds}")
    ids = sorted(set(group_ids))
    if len(ids) < int(n_folds):
        raise ValueError(f"{len(ids)} groups cannot fill {n_folds} folds")
    rng = np.random.default_rng(int(seed))
    order = [ids[int(i)] for i in rng.permutation(len(ids))]
    folds: list[list[str]] = [[] for _ in range(int(n_folds))]
    for i, gid in enumerate(order):
        folds[i % int(n_folds)].append(gid)
    return folds


def choose_groups(groups: list[int], frac: float, seed: int, widths: dict | None = None) -> set[int]:
    """A nested seeded fraction of group ids. One group is always kept.

    ``widths`` maps a group id to a section width. The prefix then keeps at
    least one group of every width in the eligible pool.
    """
    if frac <= 0 or frac > 1:
        raise ValueError(f"group fraction must be in (0, 1], got {frac}")
    return set(nested_prefix([int(g) for g in groups], float(frac), int(seed), widths=widths))


def training_records(
    cache: Cache, stage: int, split: str = "train", *,
    iid_group_frac: float = 1.0, iid_seeds_per_group: int | None = None, seed: int = 0,
    manifold_group_frac: float = 1.0, manifold_exclude: set[str] | None = None,
    include_pair_val: bool = False,
) -> list[dict]:
    """Rows the sampler draws from. Pairs share ``draw`` so they see one window.

    Stage 2 replays IID only and adds the manifold. Unified training uses IID,
    the manifold, and a repeated-column slice whose residual is exactly zero.
    Dipping and three-layer are held out and scored only. Cosine-bowl troughs
    are already absent from the cache.
    """
    if split in BLOCKED_TRAIN_SPLITS:
        raise ValueError(f"refusing to train on split {split!r}")
    records: list[dict] = []
    if stage == STAGE_UNIFIED:
        families = ("iid",)
    elif stage < 2:
        families = ("iid", "ood_dipping", "ood_three_layer")
    else:
        families = ("iid",)
    by_group: dict[tuple[int, int], list[int]] = {}
    iid_keep = None
    if iid_group_frac < 1.0 or iid_seeds_per_group is not None:
        grouped: dict[int, list[int]] = {}
        for flat in cache.runs(split, ("iid",)):
            camp, j = cache.campaign(int(flat))
            grouped.setdefault(int(camp.group[j]), []).append(int(flat))
        iid_keep = choose_iid_flats(grouped, iid_group_frac, iid_seeds_per_group, seed)
    for flat in cache.runs(split, families):
        if iid_keep is not None and cache.campaigns[cache.run_of[int(flat)][0]].name == "iid" and int(flat) not in iid_keep:
            continue
        camp, j = cache.campaign(flat)
        rec = {
            "kind": "run", "index": int(flat), "draw": len(records), "fixed": False,
            "stride": 0, "allow_mirror": True, "family": camp.name, "group": int(camp.group[j]),
            "pair": False,
        }
        records.append(rec)
        by_group.setdefault((FAMILY[camp.name], int(camp.group[j])), []).append(len(records) - 1)
    pair_of: dict[int, int] = {}
    for members in by_group.values():
        if len(members) < 2:
            continue
        a, b = members[0], members[1]
        shared = records[a]["draw"]
        records[b]["draw"] = shared
        records[a]["pair"] = records[b]["pair"] = True
        pair_of[a] = b
    if stage >= 2 or stage == STAGE_UNIFIED:
        section_rows = []
        source_index = list(cache.section_indices(split))
        if include_pair_val and split == "train":
            pairs = set(manifold_pair_group_ids(cache))
            for j in cache.section_indices("val"):
                if cache.sections[int(j)].group_id in pairs:
                    source_index.append(int(j))
        for j in source_index:
            sec = cache.sections[j]
            if sec.split in BLOCKED_TRAIN_SPLITS or sec.split == "blind":
                continue
            if manifold_exclude and sec.group_id in manifold_exclude:
                continue
            section_rows.append((int(j), sec))
        widths = {int(sec.group): int(sec.lx) for _, sec in section_rows}
        keep_groups = choose_groups([int(sec.group) for _, sec in section_rows], manifold_group_frac, seed, widths=widths)
        kept_rows = [(j, sec) for j, sec in section_rows if int(sec.group) in keep_groups]
        for j, sec in kept_rows:
            records.append({
                "kind": "section", "index": int(j), "draw": len(records), "fixed": True,
                "width": sec.lx, "start": 0, "stride": TRAIN_STRIDE_M, "allow_mirror": True,
                "family": "manifold", "group": sec.group, "stratum": sec.stratum, "pair": False,
                "lx": sec.lx,
                "near_1d": lateral_log_std(sec.vs) < NEAR_1D_LOG_STD,
                "f1_hz": float(sec.f1_hz),
                "f2_hz": float(sec.f2_hz),
            })
        sec_groups: dict[int, list[int]] = {}
        for i, rec in enumerate(records):
            if rec["kind"] == "section":
                sec_groups.setdefault(int(rec["group"]), []).append(i)
        for members in sec_groups.values():
            if len(members) < 2 or records[members[0]].get("stratum") == "matched":
                continue
            a, b = members[0], members[1]
            records[b]["draw"] = records[a]["draw"]
            records[a]["pair"] = records[b]["pair"] = True
    if stage == STAGE_UNIFIED:
        extra = _cov0_records(cache, split)
        for rec in extra:
            rec["draw"] = len(records)
            records.append(rec)
    return records


def eval_records(
    cache: Cache, split: str, *, iid_limit: int | None = None, positions: int = 1,
    families: tuple[str, ...] = ("iid", "ood_dipping", "ood_three_layer"),
    section_stride: int = 5, widths: tuple[int, ...] = EVAL_WIDTHS,
    extra_section_splits: tuple[str, ...] = (),
    section_group_ids: set[str] | None = None,
) -> list[dict]:
    """Fixed windows. Runs are centered. Sections are scored at their own width and at crops that fit.

    ``extra_section_splits`` adds sections that are logged and not part of the
    requested run split. Blind manifold sections are the reported test.
    """
    out = []
    n_runs = 0
    for flat in cache.runs(split, families):
        if iid_limit is not None and n_runs >= iid_limit:
            break
        n_runs += 1
        camp, _ = cache.campaign(flat)
        for w in IID_EVAL_WIDTHS:
            out.append({
                "kind": "run", "index": int(flat), "draw": 0, "fixed": True, "allow_mirror": False,
                "width": int(w), "start": IID_MESH_M // 2 - int(w) // 2, "stride": 0, "family": camp.name,
            })
    if section_group_ids is None:
        section_splits: list[str] = []
        for sec_split in (split, *extra_section_splits):
            if sec_split not in section_splits:
                section_splits.append(sec_split)
        for sec_split in section_splits:
            for j in cache.section_indices(sec_split):
                _append_section(out, cache.sections[j], j, positions, section_stride, widths)
        return out
    for sec_split in ("train", "val"):
        for j in cache.section_indices(sec_split):
            sec = cache.sections[int(j)]
            if sec.group_id in section_group_ids:
                _append_section(out, sec, int(j), positions, section_stride, widths)
    return out


def _append_section(out, sec, j, positions, section_stride, widths) -> None:
    native = min(sec.lx, W_MAX_M)
    out.append({
        "kind": "section", "index": int(j), "draw": 0, "fixed": True, "allow_mirror": False,
        "width": native, "start": 0, "stride": section_stride, "family": "manifold",
        "stratum": sec.stratum, "group": sec.group,
    })
    if positions > 1 and sec.lx >= 1000:
        for w in widths:
            if w >= sec.lx or w < W_MIN_M:
                continue
            starts = np.linspace(0, sec.lx - w, positions).round().astype(int)
            for s in np.unique(starts):
                out.append({
                    "kind": "section", "index": int(j), "draw": 0, "fixed": True, "allow_mirror": False,
                    "width": int(w), "start": int(s), "stride": section_stride, "family": "manifold",
                    "stratum": sec.stratum,
                })

