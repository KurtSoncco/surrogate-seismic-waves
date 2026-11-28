"""Station-aligned DeltaPhi retrieval for the manifold residual.

A query is a descriptor at one station: depth bins of the column under that
station and of the columns 250 m and 500 m away, plus log f1, log f2 and the
lateral log-Vs scatter. Retrieval is cosine similarity inside the training
archive. Rows that share the query's group id are removed, so a split cannot
see its twin. Training draws one of the top-K neighbours; inference uses the
top one. The head predicts delta_g = g - g_k and is zero at init, so a
neighbour with g_k = 0 stays a near-1D prediction until the head moves.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

OFFSETS_M = (0.0, -250.0, 250.0, -500.0, 500.0)
DEPTH_BINS = 16


def _column_at(vs: np.ndarray, x_m: float) -> np.ndarray:
    x = float(np.clip(x_m, 0.0, vs.shape[1] - 1))
    i0 = int(np.floor(x))
    i1 = min(i0 + 1, vs.shape[1] - 1)
    t = x - i0
    return (1.0 - t) * np.asarray(vs[:, i0], dtype=np.float64) + t * np.asarray(vs[:, i1], dtype=np.float64)


def _bin_column(column: np.ndarray, n_bins: int = DEPTH_BINS) -> np.ndarray:
    if column.size == 0:
        return np.zeros(n_bins, dtype=np.float64)
    edges = np.linspace(0, column.size, n_bins + 1).astype(int)
    out = np.empty(n_bins, dtype=np.float64)
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        b = max(b, a + 1)
        out[i] = np.log(np.maximum(column[a:b].mean(), 1e-6) / 400.0)
    return out


def station_descriptor(
    vs: np.ndarray,
    x_m: float,
    *,
    f1_hz: float,
    f2_hz: float,
    lateral_log_std: float,
) -> np.ndarray:
    """One vector. Offsets that fall outside the section repeat the edge column."""
    pieces = [_bin_column(_column_at(vs, x_m + offset)) for offset in OFFSETS_M]
    pieces.append(np.array([
        np.log(max(float(f1_hz), 1e-6)),
        np.log(max(float(f2_hz), 1e-6)),
        float(lateral_log_std),
    ], dtype=np.float64))
    return np.concatenate(pieces)


def _unit(rows: np.ndarray) -> np.ndarray:
    scale = np.linalg.norm(rows, axis=-1, keepdims=True)
    return rows / np.maximum(scale, 1e-12)


class RetrievalIndex:
    """Cosine archive. ``groups[i]`` equal to the query group is never returned."""

    def __init__(self, descriptors: np.ndarray, curves: np.ndarray, groups: np.ndarray):
        desc = np.asarray(descriptors, dtype=np.float64)
        curves = np.asarray(curves, dtype=np.float64)
        groups = np.asarray(groups)
        if desc.shape[0] != curves.shape[0] or desc.shape[0] != groups.shape[0]:
            raise ValueError("descriptors, curves and groups must have the same length")
        self.desc = _unit(desc)
        self.curves = curves
        self.groups = groups

    def retrieve(
        self,
        descriptor: np.ndarray,
        group,
        k: int,
        *,
        train: bool,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """Return (curve, similarity, top-k indices). Training picks one of the top k."""
        query = _unit(np.asarray(descriptor, dtype=np.float64).reshape(1, -1))[0]
        sim = self.desc @ query
        sim = np.where(self.groups == group, -np.inf, sim)
        valid = np.isfinite(sim)
        if not np.any(valid):
            curve = np.zeros(self.curves.shape[1], dtype=np.float64)
            return curve, 0.0, np.zeros(0, dtype=np.int64)
        order = np.argsort(sim)[::-1]
        order = order[np.isfinite(sim[order])]
        take = order[: max(int(k), 1)]
        if take.size == 0:
            curve = np.zeros(self.curves.shape[1], dtype=np.float64)
            return curve, 0.0, take
        if train and take.size > 1:
            rng = np.random.default_rng() if rng is None else rng
            chosen = int(take[int(rng.integers(0, take.size))])
        else:
            chosen = int(take[0])
        return self.curves[chosen], float(sim[chosen]), take


def descriptor_from_canvas(vs_log: np.ndarray, column: float, f1_hz: float, f2_hz: float) -> np.ndarray:
    """Descriptor from one canvas column of log(Vs/400). ``vs_log`` is (depth, width)."""
    vs = 400.0 * np.exp(np.asarray(vs_log, dtype=np.float64))
    lateral = float(np.log(np.maximum(vs, 1.0)).std(axis=1).mean())
    return station_descriptor(vs, column, f1_hz=f1_hz, f2_hz=f2_hz, lateral_log_std=lateral)


def index_from_cache(cache) -> RetrievalIndex | None:
    """Training manifold stations at the centre recorder. Troughs are already out of the cache."""
    from data import lateral_log_std

    desc, curves, groups = [], [], []
    for sec in cache.sections:
        if sec.split != "train":
            continue
        centre = int(sec.log_tf.shape[0] // 2)
        g = np.asarray(sec.log_tf[centre], dtype=np.float64) - np.asarray(sec.log_tf1d[centre], dtype=np.float64)
        x = float(sec.x_m[centre]) if sec.x_m.shape[0] else 0.0
        desc.append(station_descriptor(
            np.asarray(sec.vs), x, f1_hz=float(sec.f1_hz), f2_hz=float(sec.f2_hz),
            lateral_log_std=lateral_log_std(sec.vs),
        ))
        curves.append(g)
        groups.append(int(sec.group))
    if not desc:
        return None
    return RetrievalIndex(np.stack(desc), np.stack(curves), np.asarray(groups))


def retrieve_batch(index: RetrievalIndex, x, query, pred, info, *, train: bool, rng=None):
    """One neighbour residual per sample, shaped like ``pred`` and broadcast across its stations."""
    from encoding import CH_LOG_F1, CH_LOG_F2, F_REF_HZ

    vs = x[:, 0].detach().float().cpu().numpy()
    q = query.detach().float().cpu().numpy()
    info_c = info.detach().cpu().numpy()
    g_k = torch.zeros_like(pred)
    sim = torch.ones(pred.shape[0], pred.shape[1], 1, device=pred.device, dtype=pred.dtype)
    for b in range(pred.shape[0]):
        f1b = float(np.exp(x[b, CH_LOG_F1, 0, 0].detach().float().cpu())) * F_REF_HZ
        f2b = float(np.exp(x[b, CH_LOG_F2, 0, 0].detach().float().cpu())) * F_REF_HZ
        descriptor = descriptor_from_canvas(vs[b], float(q[b, 0]), f1b, f2b)
        curve, similarity, _top = index.retrieve(descriptor, int(info_c[b, 5]), 20, train=train, rng=rng)
        g_k[b] = torch.as_tensor(curve, device=pred.device, dtype=pred.dtype)
        sim[b] = float(similarity)
    return g_k, sim


class DeltaPhiHead(nn.Module):
    """Predict delta_g from the station latent, the retrieved residual, and the similarity.

    The last layer is zero, so g_hat = g_k until training moves it. A retrieved
    near-1D neighbour (g_k = 0) therefore starts at a zero residual.
    """

    def __init__(self, latent: int, n_freq: int, hidden: int = 64):
        super().__init__()
        self.curve = nn.Linear(n_freq, latent)
        self.out = nn.Linear(3 * latent + 1, n_freq)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, latent: torch.Tensor, g_k: torch.Tensor, sim: torch.Tensor) -> torch.Tensor:
        """latent (..., C), g_k (..., F), sim (..., 1) -> delta_g (..., F)."""
        code = torch.cat([latent, self.curve(g_k), latent * self.curve(g_k), sim], dim=-1)
        return self.out(code)
