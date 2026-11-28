"""LOGLO-POD for windows of any width: one architecture for both encodings.

lift (narrow, full grid) -> mask-gated depth collapse -> expand -> FiLM on
site groups -> DualPathLOGLOStack along x -> shared log-POD head evaluated
only at the station columns of each sample.

The collapse is a per-row weighted average over mesh rows only, so half-space
padding never enters the surface token and row i is the same depth for every
mesh. FiLM groups are averaged over the window's own columns, so canvas
padding (metric encoding) does not move them.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GIFNO_DIR = Path(__file__).resolve().parents[1] / "GIFNO"
if str(GIFNO_DIR) not in sys.path:
    sys.path.append(str(GIFNO_DIR))

from spectral_layers import DualPathLOGLOStack  # noqa: E402

from encoding import (  # noqa: E402
    CH_LATERAL,
    CH_LOG_F0,
    CH_LOG_F1,
    CH_LOG_F2,
    CH_LOG_H,
    CH_LOG_IMPEDANCE,
    CH_LOG_ZETA,
    CH_MASK,
    CH_X,
    CH_Z,
    FILM_CHANNELS,
    FILM_ONLY,
    N_CHANNELS,
    grid_shape,
)

# Odd metre kernels. A column's FiLM reads the interior inside each radius, not the canvas origin.
LOCAL_KERNELS_M = (251, 501, 1001)
LOCAL_STAT_CHANNELS = (CH_LOG_F0, CH_LOG_IMPEDANCE, CH_LOG_ZETA, CH_LOG_H)

# NEHRP Vs30 boundaries, C/D and B/C, used here as crossings of Vs(z).
CONTACT_VS_MPS = (760.0, 360.0)
CONTACT_RUN_M = 4
CONTACT_DEPTH_M = 256.0
VS_REF_MPS = 400.0
# depth760, depth360, shallow depth, shallow Δlog, deep depth, deep Δlog, crossed760, crossed360.
CONTACT_CHANNELS = 8


def _log_vs_threshold(vs_mps: float) -> float:
    return math.log(float(vs_mps) / VS_REF_MPS)


def contact_raw(log_vs: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    """(B, H, W) log(Vs/400) and a row mask -> (B, 8, 1, W) contact features.

    A threshold crossing is the first depth that stays above the threshold for
    4 m, and only where the row above was not already above it. A column that
    never holds that run, or that starts above the threshold, gets depth 1 and
    a crossed flag of 0. The two jumps are the largest non-overlapping 4 m
    log-Vs differences, stored shallow then deep.
    """
    if log_vs.ndim != 3 or observed.shape != log_vs.shape:
        raise ValueError(f"log_vs {tuple(log_vs.shape)} and observed {tuple(observed.shape)} must be (B, H, W)")
    batch, height, width = log_vs.shape
    device = log_vs.device
    dtype = log_vs.dtype
    raw = log_vs.new_zeros(batch, CONTACT_CHANNELS, width)
    raw[:, 0, :] = 1.0
    raw[:, 1, :] = 1.0
    raw[:, 2, :] = 1.0
    raw[:, 4, :] = 1.0
    run = CONTACT_RUN_M
    if height < run + 1:
        return raw.unsqueeze(2)
    observed = observed.to(dtype=torch.bool)
    for slot, vs_mps in enumerate(CONTACT_VS_MPS):
        above = (log_vs > _log_vs_threshold(vs_mps)) & observed
        persist = above[:, :-3] & above[:, 1:-2] & above[:, 2:-1] & above[:, 3:]
        started = persist[:, 1:]
        previous = above[:, : started.shape[1]]
        transition = started & ~previous
        found = transition.any(dim=1)
        depth_row = transition.to(dtype=torch.float32).argmax(dim=1) + 1
        depth = torch.where(found, depth_row.to(dtype) / CONTACT_DEPTH_M, torch.ones_like(depth_row, dtype=dtype))
        raw[:, slot, :] = depth
        raw[:, 6 + slot, :] = found.to(dtype)
    delta = log_vs[:, run:] - log_vs[:, :-run]
    valid = observed[:, run:] & observed[:, :-run]
    magnitude = delta.abs().masked_fill(~valid, -1.0)
    first = magnitude.argmax(dim=1)
    has_first = magnitude.amax(dim=1) > -0.5
    rows = torch.arange(magnitude.shape[1], device=device)
    near = (rows[None, :, None] - first[:, None, :]).abs() < run
    second_mag = magnitude.masked_fill(near, -1.0)
    second = second_mag.argmax(dim=1)
    has_second = second_mag.amax(dim=1) > -0.5
    d_first = delta.gather(1, first.unsqueeze(1)).squeeze(1)
    d_second = delta.gather(1, second.unsqueeze(1)).squeeze(1)
    swap = has_second & (second < first)
    shallow_i = torch.where(swap, second, first)
    deep_i = torch.where(swap, first, second)
    shallow_d = torch.where(swap, d_second, d_first)
    deep_d = torch.where(swap, d_first, d_second)
    one = torch.ones_like(shallow_d)
    zero = torch.zeros_like(shallow_d)
    mid = (CONTACT_RUN_M / 2.0) / CONTACT_DEPTH_M
    raw[:, 2, :] = torch.where(has_first, shallow_i.to(dtype) / CONTACT_DEPTH_M + mid, one)
    raw[:, 3, :] = torch.where(has_first, shallow_d, zero)
    raw[:, 4, :] = torch.where(has_second, deep_i.to(dtype) / CONTACT_DEPTH_M + mid, one)
    raw[:, 5, :] = torch.where(has_second, deep_d, zero)
    return raw.unsqueeze(2)


def fit_contact_whitening(cache, column_stride: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of the eight raw contact channels.

    The sample is the same one the joint column PCA uses: training IID runs
    and training manifold sections. Statistics are frozen before training.
    """
    # The two crossed flags stay 0/1. On these profiles both are almost always 1,
    # so whitening them would divide by a zero standard deviation.
    kept = 6
    total = np.zeros(kept, dtype=np.float64)
    total_sq = np.zeros(kept, dtype=np.float64)
    count = 0

    def consume(vs: np.ndarray) -> None:
        nonlocal count
        field = np.asarray(vs, dtype=np.float64)
        if field.ndim != 2 or field.shape[0] < CONTACT_RUN_M + 1 or field.shape[1] < 1:
            return
        cols = np.arange(0, field.shape[1], max(int(column_stride), 1))
        taken = field[:, cols]
        log_vs = np.log(np.maximum(taken, 1e-6) / VS_REF_MPS).astype(np.float32)
        observed = np.ones_like(log_vs, dtype=bool)
        raw = contact_raw(torch.from_numpy(log_vs).unsqueeze(0), torch.from_numpy(observed).unsqueeze(0))
        values = raw[0, :kept, 0, :].detach().numpy().astype(np.float64)
        total[:] += values.sum(axis=1)
        total_sq[:] += np.square(values).sum(axis=1)
        count += int(values.shape[1])

    runs = cache.runs("train", ("iid",))
    for n_done, flat in enumerate(runs, start=1):
        camp, j = cache.campaign(int(flat))
        n_active = int(camp.n_active[j])
        consume(np.asarray(camp.vs[j, :n_active]))
        if n_done % 1000 == 0:
            print(f"contact whitening runs {n_done}/{len(runs)}", flush=True)
    for j in cache.section_indices("train"):
        consume(np.asarray(cache.sections[int(j)].vs))
    if count < 2:
        raise RuntimeError("contact whitening saw fewer than two columns")
    mean = total / count
    var = total_sq / count - np.square(mean)
    std = np.sqrt(np.maximum(var, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)

DEFAULT_MODES_X = {"metric": 96, "reference": 32}


class MaskedDepthCollapse(nn.Module):
    """(B, C, H, W) -> (B, C, 1, W): sum_z w[c, z] m[z, x] h[c, z, x] / sum_z m[z, x]."""

    def __init__(self, channels: int, depth: int):
        super().__init__()
        self.weight = nn.Parameter(1.0 + 0.1 * torch.randn(channels, depth))

    def forward(self, h: torch.Tensor, row_mask: torch.Tensor) -> torch.Tensor:
        num = torch.einsum("bchw,ch,bhw->bcw", h, self.weight.to(h.dtype), row_mask.to(h.dtype))
        den = row_mask.sum(dim=1).clamp_min(1.0).to(h.dtype)
        return (num / den[:, None, :]).unsqueeze(2)


class ColumnDepthAutoencoder(nn.Module):
    """Mask-gated column encoder. No parameter has length H.

    Each row is a token. A positive score from that token replaces the static
    row weight. The surface code is sum(score * token * mask) / sum(mask), so a
    half-space row contributes nothing. The decoder rebuilds the lifted row
    from the surface code and that row's depth channel.
    """

    def __init__(self, channels: int, code_dim: int | None = None):
        super().__init__()
        # Default code width equals the channel count, so the trunk's state dict is unchanged.
        # The soil-profile fit uses a wider code: two input channels cannot hold a contact depth.
        width = int(channels if code_dim is None else code_dim)
        self.code_dim = width
        self.lift = None if width == channels else nn.Conv2d(channels, width, 1)
        self.row = nn.Sequential(
            nn.Conv2d(width, width, 1),
            nn.GELU(),
            nn.Conv2d(width, width, 1),
        )
        nn.init.zeros_(self.row[-1].weight)
        nn.init.zeros_(self.row[-1].bias)
        self.score = nn.Conv2d(width, width, 1)
        nn.init.zeros_(self.score.weight)
        nn.init.constant_(self.score.bias, math.log(math.e - 1.0))
        self.decode = nn.Sequential(
            nn.Conv2d(width + 1, width, 1),
            nn.GELU(),
            nn.Conv2d(width, channels, 1),
        )
        nn.init.zeros_(self.decode[-1].weight)
        nn.init.zeros_(self.decode[-1].bias)

    def forward(
        self, h: torch.Tensor, row_mask: torch.Tensor, depth: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lifted = h if self.lift is None else self.lift(h)
        token = lifted + self.row(lifted)
        score = F.softplus(self.score(token))
        mask = row_mask.to(dtype=h.dtype)
        num = (score * token * mask[:, None]).sum(dim=2)
        den = mask.sum(dim=1).clamp_min(1.0)
        code = num / den[:, None, :]
        surface = code.unsqueeze(2)
        depth_b = depth.to(dtype=h.dtype)
        recon_in = torch.cat([surface.expand(-1, -1, h.shape[2], -1), depth_b], dim=1)
        return surface, self.decode(recon_in)


class SpectralDecoder(nn.Module):
    """Coordinate MLP from a latent and log(f / 1 Hz) to one log-spectrum value."""

    def __init__(self, channels: int, hidden: int, n_bands: int = 4):
        super().__init__()
        self.n_bands = int(n_bands)
        self.net = nn.Sequential(
            nn.Linear(channels + 1 + 2 * self.n_bands, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, latent: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
        log_f = freq.to(dtype=latent.dtype, device=latent.device).clamp_min(1e-6).log()
        bands = torch.arange(1, self.n_bands + 1, device=latent.device, dtype=latent.dtype)
        ang = log_f[:, None] * bands[None, :]
        feat = torch.cat([log_f[:, None], ang.sin(), ang.cos()], dim=-1)
        b, k, _ = latent.shape
        outs = []
        for i in range(0, feat.shape[0], 64):
            fr = feat[i : i + 64]
            lat = latent[:, :, None, :].expand(b, k, fr.shape[0], latent.shape[-1])
            inp = torch.cat([lat, fr[None, None].expand(b, k, fr.shape[0], fr.shape[-1])], dim=-1)
            outs.append(self.net(inp).squeeze(-1))
        return torch.cat(outs, dim=-1)


def frequency_neighbor_mask(n_bins: int, knn: int, block: int, stride: int) -> torch.Tensor:
    """Binary output mask (n_bins, M2). Each bin connects to the hidden blocks of its nearest bins.

    Hidden block ``j`` occupies indices ``[j * stride, j * stride + block)``. The width is
    ``block + stride * (n_bins - 1)``, the 1D form of Kim and Choi Eqs. (B.13)–(B.16).
    """
    if n_bins < 1 or knn < 1 or block < 1 or stride < 1:
        raise ValueError(f"need positive sizes, got n_bins={n_bins} knn={knn} block={block} stride={stride}")
    if knn > n_bins:
        raise ValueError(f"knn={knn} exceeds n_bins={n_bins}")
    width = block + stride * (n_bins - 1)
    mask = torch.zeros(n_bins, width)
    index = np.arange(n_bins)
    for i in range(n_bins):
        neighbors = np.lexsort((index, np.abs(index - i)))[:knn]
        for j in neighbors:
            start = int(j) * stride
            mask[i, start : start + block] = 1.0
    return mask


class SparseWindowAutoencoder(nn.Module):
    """Shallow autoencoder for one frequency window.

    The encoder is an MLP with SiLU hidden layers and no bias on the code.
    The decoder has one SiLU hidden layer and a bias-free output whose weights
    are masked to the k nearest bins on the frequency line.
    """

    def __init__(
        self,
        n_bins: int,
        n_loc: int,
        *,
        knn: int = 5,
        block: int = 4,
        stride: int = 1,
        hidden: tuple[int, ...] = (64, 32),
    ):
        super().__init__()
        mask = frequency_neighbor_mask(n_bins, knn, block, stride)
        self.n_bins = int(n_bins)
        self.n_loc = int(n_loc)
        layers: list[nn.Module] = []
        width = n_bins
        for hid in hidden:
            layers.append(nn.Linear(width, hid))
            layers.append(nn.SiLU())
            width = hid
        layers.append(nn.Linear(width, n_loc, bias=False))
        self.encoder = nn.Sequential(*layers)
        hidden_width = int(mask.shape[1])
        self.lift = nn.Linear(n_loc, hidden_width)
        self.out_weight = nn.Parameter(torch.empty(n_bins, hidden_width))
        nn.init.kaiming_uniform_(self.out_weight, a=math.sqrt(5))
        self.register_buffer("mask", mask)
        with torch.no_grad():
            self.out_weight.mul_(self.mask)

    def encode(self, window: torch.Tensor) -> torch.Tensor:
        return self.encoder(window)

    def decode(self, code: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.lift(code))
        weight = self.out_weight * self.mask
        return F.linear(hidden, weight)

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(window))


class CurveEncoder(nn.Module):
    """MLP from a residual curve (B, F) to k codes. The curve is the only input."""

    def __init__(self, n_freq: int, k: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_freq, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, k),
        )

    def forward(self, curves: torch.Tensor) -> torch.Tensor:
        return self.net(curves)


class SiteFiLM(nn.Module):
    def __init__(self, n_groups: int, channels: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_groups, hidden), nn.GELU(), nn.Linear(hidden, 2 * channels))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.net(cond).chunk(2, dim=-1)
        return h * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]


class LocalContextFiLM(nn.Module):
    """Per-column FiLM. ``cond`` is (B, W, G) and ``h`` is (B, C, 1, W)."""

    def __init__(self, n_groups: int, channels: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_groups, hidden), nn.GELU(), nn.Linear(hidden, 2 * channels))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.net(cond).chunk(2, dim=-1)
        gamma = gamma.permute(0, 2, 1).unsqueeze(2)
        beta = beta.permute(0, 2, 1).unsqueeze(2)
        return h * (1.0 + gamma) + beta


_BOX_KERNELS: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def _box_kernels(channels: int, kernel: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    key = (channels, kernel, device.type, device.index, dtype)
    found = _BOX_KERNELS.get(key)
    if found is None:
        found = (
            torch.ones(channels, 1, kernel, device=device, dtype=dtype),
            torch.ones(1, 1, kernel, device=device, dtype=dtype),
        )
        _BOX_KERNELS[key] = found
    return found


def masked_box_average(signal: torch.Tensor, mask: torch.Tensor, kernel: int) -> torch.Tensor:
    """Average ``signal`` (B, C, W) over an odd window, weighted by ``mask`` (B, 1, W)."""
    if kernel % 2 == 0:
        raise ValueError(f"local kernel must be odd, got {kernel}")
    weight, den_w = _box_kernels(int(signal.shape[1]), kernel, signal.device, signal.dtype)
    num = F.conv1d(signal * mask, weight, padding=kernel // 2, groups=int(signal.shape[1]))
    den = F.conv1d(mask, den_w, padding=kernel // 2)
    width = signal.shape[-1]
    return num[..., :width] / den[..., :width].clamp_min(1.0)


class VarDomainLOGLO(nn.Module):
    def __init__(
        self,
        encoding: str,
        pod_modes: np.ndarray,
        pod_mean: np.ndarray,
        *,
        n_modes_x: int | None = None,
        narrow: int = 32,
        latent: int = 128,
        n_layers: int = 5,
        patch_w: int = 20,
        hfp: int = 4,
        hfp_kernel: int | None = None,
        hfp_stride: int | None = None,
        local_min_wavelength: float = 0.0,
        hf_noise_alpha: float = 0.025,
        branch_hidden: int = 256,
        film_hidden: int = 64,
        log_clamp: float = 20.0,
        pointwise: bool = False,
        readout: str = "pod",
        depth_ae: bool = False,
        recon_weight: float = 0.1,
        depth_pca: bool = False,
        pca_mean: np.ndarray | None = None,
        pca_components: np.ndarray | None = None,
        translation_equivariant: bool = False,
        contact_features: bool = False,
        contact_mean: np.ndarray | None = None,
        contact_std: np.ndarray | None = None,
        input_weight: np.ndarray | None = None,
    ):
        super().__init__()
        nz, nx = grid_shape(encoding)
        if nx % patch_w:
            raise ValueError(f"grid width {nx} must be a multiple of patch_w={patch_w}")
        if readout not in ("pod", "nonlinear"):
            raise ValueError(f"readout must be 'pod' or 'nonlinear', got {readout!r}")
        self.encoding = encoding
        self.log_clamp = float(log_clamp)
        self.readout_kind = readout
        self.depth_ae = bool(depth_ae)
        self.depth_pca = bool(depth_pca)
        self.translation_equivariant = bool(translation_equivariant)
        if self.depth_ae and self.depth_pca:
            raise ValueError("depth autoencoder and frozen PCA-16 are different depth codes")
        self.recon_weight = float(recon_weight)
        self._recon: torch.Tensor | None = None
        self._lifted: torch.Tensor | None = None
        self._row_mask: torch.Tensor | None = None
        dropped = set(FILM_ONLY)
        if self.translation_equivariant:
            dropped.add(CH_X)
        lift_idx = [c for c in range(N_CHANNELS) if c not in dropped]
        self.register_buffer("lift_idx", torch.tensor(lift_idx), persistent=False)
        self.register_buffer("film_idx", torch.tensor(FILM_CHANNELS), persistent=False)
        self.column_pca: FrozenColumnPCA | None
        self.pca_to_narrow: nn.Sequential | None
        if self.depth_pca:
            if pca_mean is None or pca_components is None:
                raise ValueError("frozen PCA depth code needs pca_mean and pca_components")
            self.column_pca = (
                WeightedColumnPOD(pca_mean, pca_components, input_weight)
                if input_weight is not None else FrozenColumnPCA(pca_mean, pca_components)
            )
            self.coupled: CoupledPODMap | None = None
            rank = int(self.column_pca.components.shape[0])
            self.pca_to_narrow = nn.Sequential(nn.Conv2d(rank, narrow, 1), nn.GELU())
            self.lift = nn.Identity()
            self.collapse = nn.Identity()
        else:
            self.column_pca = None
            self.coupled = None
            self.pca_to_narrow = None
            self.lift = nn.Sequential(nn.Conv2d(len(lift_idx), narrow, 1), nn.GELU())
            self.collapse = ColumnDepthAutoencoder(narrow) if self.depth_ae else MaskedDepthCollapse(narrow, nz)
        self.expand = nn.Sequential(nn.Conv2d(narrow, latent, 1), nn.GELU())
        if self.translation_equivariant:
            n_local = len(LOCAL_STAT_CHANNELS) * len(LOCAL_KERNELS_M) + 2
            self.film = LocalContextFiLM(n_local, latent, film_hidden)
        else:
            self.film = SiteFiLM(len(FILM_CHANNELS), latent, film_hidden)
        kernel = int(hfp if hfp_kernel is None else hfp_kernel)
        stride = int(hfp if hfp_stride is None else hfp_stride)
        self.hfp_kernel = kernel
        self.hfp_stride = stride
        self.local_min_wavelength = float(local_min_wavelength)
        self.encoder = DualPathLOGLOStack(
            n_modes=(1, int(n_modes_x or DEFAULT_MODES_X[encoding])),
            channels=latent,
            n_layers=n_layers,
            patch_size=(1, patch_w),
            hfp_kernel=kernel,
            hfp_stride=stride,
            hf_noise_alpha=hf_noise_alpha,
            local_min_wavelength=self.local_min_wavelength,
        )
        k = int(pod_modes.shape[0])
        n_freq = int(pod_modes.shape[1])
        self.branch: nn.Sequential | None
        self.pointwise: nn.Linear | None
        self.spectral: SpectralDecoder | None
        if self.readout_kind == "nonlinear":
            self.branch = None
            self.pointwise = None
            self.spectral = SpectralDecoder(2 * latent, branch_hidden)
            self.register_buffer("freq_hz", torch.ones(n_freq, dtype=torch.float32))
        else:
            self.branch = nn.Sequential(nn.Linear(2 * latent, branch_hidden), nn.GELU(), nn.Linear(branch_hidden, k))
            self.pointwise = nn.Linear(2 * latent, n_freq) if pointwise else None
            self.spectral = None
            if self.pointwise is not None:
                nn.init.zeros_(self.pointwise.weight)
                nn.init.zeros_(self.pointwise.bias)
        self.register_buffer("pod_modes", torch.as_tensor(pod_modes, dtype=torch.float32))
        self.register_buffer("pod_mean", torch.as_tensor(pod_mean, dtype=torch.float32))
        self.register_buffer("pod_weight", torch.ones(n_freq, dtype=torch.float32))
        self.register_buffer("pod_coeff_scale", torch.ones(k, dtype=torch.float32))
        # Last, so its initialization draw does not move the layers above.
        self.contact_proj: nn.Conv2d | None
        if contact_features:
            if contact_mean is None or contact_std is None:
                raise ValueError("contact features need a frozen mean and standard deviation")
            mean_t = torch.as_tensor(np.asarray(contact_mean, dtype=np.float32))
            std_t = torch.as_tensor(np.asarray(contact_std, dtype=np.float32))
            if mean_t.shape != (6,) or std_t.shape != (6,):
                raise ValueError("contact whitening covers the six continuous channels")
            self.register_buffer("contact_mean", mean_t)
            self.register_buffer("contact_std", std_t.clamp_min(1e-6))
            self.contact_proj = nn.Conv2d(CONTACT_CHANNELS, latent, 1)
            nn.init.zeros_(self.contact_proj.weight)
            nn.init.zeros_(self.contact_proj.bias)
        else:
            self.contact_proj = None

    def contact_field(self, x: torch.Tensor) -> torch.Tensor:
        """Whitened contact channels from the canvas the network was given.

        Channel 0 is already log(Vs/400). Rows with the mask off, including the
        half-space pad, are not searched. Jitter and mirror have already been
        applied by the window builder, so this reads the field the lift sees.
        """
        height = int(x.shape[-2])
        log_vs = x[:, 0, :height, :]
        observed = x[:, CH_MASK, :height, :] > 0.5
        raw = contact_raw(log_vs, observed)
        out = raw.clone()
        mean = self.contact_mean.view(1, 6, 1, 1)
        std = self.contact_std.view(1, 6, 1, 1)
        out[:, :6] = (raw[:, :6] - mean) / std
        return out

    def site_groups(self, x: torch.Tensor) -> torch.Tensor:
        inside = (x[:, CH_LATERAL, 0, :] > 0.5).to(x.dtype)
        groups = x[:, self.film_idx, 0, :]
        return (groups * inside[:, None, :]).sum(-1) / inside.sum(-1, keepdim=True).clamp_min(1.0)

    def local_context(self, x: torch.Tensor) -> torch.Tensor:
        """(B, W, G) local averages at 250 m, 500 m and 1000 m, plus log f1 and log f2.

        The average is taken over the interior mask, so padding the canvas does
        not move the value at a station. log f1 and log f2 are constant in x.
        """
        stats = x[:, list(LOCAL_STAT_CHANNELS), 0, :]
        mask = (x[:, CH_LATERAL, 0, :] > 0.5).to(dtype=stats.dtype).unsqueeze(1)
        pieces = [masked_box_average(stats, mask, kernel) for kernel in LOCAL_KERNELS_M]
        f1 = x[:, CH_LOG_F1, 0, :].unsqueeze(1)
        f2 = x[:, CH_LOG_F2, 0, :].unsqueeze(1)
        return torch.cat([*pieces, f1, f2], dim=1).transpose(1, 2)

    def grow_modes(self, extra: torch.Tensor) -> None:
        """Append orthogonal POD rows. New branch outputs stay zero, so the old map is unchanged."""
        extra = extra.to(device=self.pod_modes.device, dtype=self.pod_modes.dtype)
        if extra.ndim != 2 or extra.shape[1] != self.pod_modes.shape[1]:
            raise ValueError(f"extra modes must be (E, F={self.pod_modes.shape[1]}), got {tuple(extra.shape)}")
        if self.branch is None:
            raise RuntimeError("nonlinear readout has no POD branch to grow")
        last = self.branch[-1]
        grown = nn.Linear(last.in_features, last.out_features + extra.shape[0])
        with torch.no_grad():
            grown.weight.zero_()
            grown.bias.zero_()
            grown.weight[: last.out_features] = last.weight
            grown.bias[: last.out_features] = last.bias
        self.branch[-1] = grown.to(last.weight.device)
        self.pod_modes = torch.cat([self.pod_modes, extra], dim=0)
        ones = torch.ones(extra.shape[0], device=self.pod_coeff_scale.device, dtype=self.pod_coeff_scale.dtype)
        self.pod_coeff_scale = torch.cat([self.pod_coeff_scale, ones], dim=0)

    def set_frequencies(self, freq: torch.Tensor) -> None:
        if self.spectral is None:
            raise RuntimeError("frequency buffer belongs to the nonlinear readout")
        freq = freq.detach().to(dtype=torch.float32, device=self.freq_hz.device).reshape(-1)
        if tuple(freq.shape) != tuple(self.freq_hz.shape):
            raise ValueError(f"frequency grid {tuple(freq.shape)} != {tuple(self.freq_hz.shape)}")
        self.freq_hz.copy_(freq)

    def reconstruction_l1(self) -> torch.Tensor:
        """Masked L1 of the lifted soil column. Rock rows are not a target."""
        if self._recon is None or self._lifted is None or self._row_mask is None:
            raise RuntimeError("depth autoencoder did not run")
        recon = self._recon.float()
        target = self._lifted.float()
        mask = self._row_mask.to(dtype=torch.float32)
        weight = mask[:, None]
        err = (recon - target).abs() * weight
        return err.sum() / (weight.sum() * recon.shape[1]).clamp_min(1.0)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        """Coefficient grid (B, M, W) or None, latent grid (B, C, W), reconstruction or None.

        The lift reads the metre coordinate, so it stays in float32. The branch
        emits whitened coefficients; ``pod_coeff_scale`` restores them. The
        nonlinear readout has no coefficient grid. The depth autoencoder's
        reconstruction is the third return and is also kept for ``reconstruction_l1``.
        """
        if self.column_pca is not None:
            with torch.autocast(device_type=x.device.type, enabled=False):
                codes = self.column_pca(x.float())
                self._input_code = codes
                surface = self.pca_to_narrow(codes)
            recon = None
            self._recon = None
            self._lifted = None
            self._row_mask = None
        else:
            with torch.autocast(device_type=x.device.type, enabled=False):
                h = self.lift(x.index_select(1, self.lift_idx).float())
            row_mask = x[:, CH_MASK]
        if self.column_pca is None and self.depth_ae:
            with torch.autocast(device_type=x.device.type, enabled=False):
                surface, recon = self.collapse(h, row_mask, x[:, CH_Z : CH_Z + 1].float())
            self._recon = recon
            self._lifted = h
            self._row_mask = row_mask
        elif self.column_pca is None:
            surface = self.collapse(h, row_mask)
            recon = None
            self._recon = None
            self._lifted = None
            self._row_mask = None
        cond = self.local_context(x) if self.translation_equivariant else self.site_groups(x)
        lifted = self.expand(surface)
        if self.contact_proj is not None:
            lifted = lifted + self.contact_proj(self.contact_field(x.float()).to(dtype=lifted.dtype))
        h = self.film(lifted, cond)
        x_global, x_local = self.encoder(h)
        latent = torch.cat([x_global[:, :, 0, :], x_local[:, :, 0, :]], dim=1)
        with torch.autocast(device_type=x.device.type, enabled=False):
            latent = latent.float()
            if self.branch is None:
                return None, latent, recon
            raw = self.branch(latent.transpose(1, 2)).transpose(1, 2)
            coeff = raw * self.pod_coeff_scale[:, None]
            if self.coupled is not None:
                coeff = coeff + self.coupled(self._input_code)
        return coeff, latent, recon

    def attach_coupled(self, beta: np.ndarray, x_mean: np.ndarray, y_mean: np.ndarray) -> None:
        """Add a frozen input-only map. The trainable branch starts as a zero residual."""
        if self.column_pca is None or self.branch is None:
            raise RuntimeError("coupled POD needs a frozen column basis and a POD branch")
        self.coupled = CoupledPODMap(beta, x_mean, y_mean).to(self.pod_modes.device)
        last = self.branch[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def readout(
        self, coeff_grid: torch.Tensor | None, latent: torch.Tensor, query: torch.Tensor, n_keep: int | None = None,
    ) -> torch.Tensor:
        """Sample the latent at float columns and decode the log spectrum."""
        if query.ndim == 1:
            query = query.unsqueeze(0)
        if self.spectral is not None:
            with torch.autocast(device_type=latent.device.type, enabled=False):
                sampled = sample_grid(latent, query)
                log_tf = self.spectral(sampled, self.freq_hz)
                return log_tf.clamp(-self.log_clamp, self.log_clamp)
        if coeff_grid is None:
            raise RuntimeError("POD readout needs a coefficient grid")
        with torch.autocast(device_type=coeff_grid.device.type, enabled=False):
            coeff = sample_grid(coeff_grid, query)
            if n_keep is not None:
                coeff = coeff.clone()
                coeff[..., int(n_keep) :] = 0
            log_tf = reconstruct_pod(self.pod_mean, coeff, self.pod_modes, self.pod_weight)
            if self.pointwise is not None:
                log_tf = log_tf + self.pointwise(sample_grid(latent, query))
            return log_tf.clamp(-self.log_clamp, self.log_clamp)

    def forward(self, x: torch.Tensor, query: torch.Tensor, n_keep: int | None = None) -> torch.Tensor:
        """x (B, C, H, W), query (B, K) float canvas columns -> POD field (B, K, F).

        The branch runs at every metre. Coefficients are then sampled linearly in
        x, so an integer metre matches the grid value and a half-metre is the
        midpoint. Queries stay in float32. The column 1D prior is added by the
        caller, not here.

        ``n_keep`` zeros POD coefficients from that index on. Stage 2 appended
        sixteen modes on the stage-1 basis, so ``n_keep=64`` is that ablation.
        """
        coeff_grid, latent, _recon = self.encode(x)
        return self.readout(coeff_grid, latent, query, n_keep)


def sample_grid(field: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """Linear sample of (B, C, W) at float columns (B, K) -> (B, K, C).

    Interpolation is float32. Casting the query to the coefficient dtype snaps
    a metre coordinate onto the bfloat16 lattice (1250 m becomes 1248 m).
    """
    with torch.autocast(device_type=field.device.type, enabled=False):
        field32 = field.float()
        query32 = query.float()
        width = field32.shape[-1]
        q = query32.clamp(0, width - 1)
        q0 = q.floor().long()
        q1 = (q0 + 1).clamp(max=width - 1)
        t = (q - q0.to(dtype=torch.float32)).unsqueeze(-1)
        b, c, _ = field32.shape
        k = q.shape[1]
        f0 = torch.gather(field32, 2, q0[:, None, :].expand(b, c, k)).transpose(1, 2)
        f1 = torch.gather(field32, 2, q1[:, None, :].expand(b, c, k)).transpose(1, 2)
        return f0 * (1.0 - t) + f1 * t


class CoupledPODMap(nn.Module):
    """Reduced-rank map from input-column codes to output POD coefficients.

    The forward pass accepts only the input code. A true residual cannot be supplied.
    """

    def __init__(self, beta: np.ndarray, x_mean: np.ndarray, y_mean: np.ndarray):
        super().__init__()
        beta_t = torch.as_tensor(np.asarray(beta, dtype=np.float32))
        x_t = torch.as_tensor(np.asarray(x_mean, dtype=np.float32))
        y_t = torch.as_tensor(np.asarray(y_mean, dtype=np.float32))
        if beta_t.ndim != 2 or x_t.shape != (beta_t.shape[0],) or y_t.shape != (beta_t.shape[1],):
            raise ValueError("coupled map shapes must be beta (in, out), x_mean (in,), y_mean (out,)")
        self.register_buffer("beta", beta_t)
        self.register_buffer("x_mean", x_t)
        self.register_buffer("y_mean", y_t)

    def forward(self, code: torch.Tensor) -> torch.Tensor:
        """(B, rank, 1, W) input codes -> (B, n_out, W) output coefficients."""
        if code.ndim != 4:
            raise ValueError(f"input code must be (B, rank, 1, W), got {tuple(code.shape)}")
        values = code.squeeze(2).transpose(1, 2)
        mapped = (values - self.x_mean) @ self.beta + self.y_mean
        return mapped.transpose(1, 2)


class WeightedColumnPOD(nn.Module):
    """Masked, depth-weighted POD of interleaved log Vs and log damping."""

    def __init__(self, mean: np.ndarray, components: np.ndarray, weight: np.ndarray):
        super().__init__()
        mean_t = torch.as_tensor(np.asarray(mean, dtype=np.float32))
        comp_t = torch.as_tensor(np.asarray(components, dtype=np.float32))
        weight_t = torch.as_tensor(np.asarray(weight, dtype=np.float32))
        if comp_t.ndim != 2 or mean_t.shape != (comp_t.shape[1],) or weight_t.shape != mean_t.shape:
            raise ValueError("weighted POD mean, components, and depth weight do not match")
        self.register_buffer("mean", mean_t)
        self.register_buffer("components", comp_t)
        self.register_buffer("weight", weight_t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height = int(self.components.shape[1] // 2)
        vs = x[:, 0, :height, :]
        zeta = x[:, 1, :height, :]
        observed = x[:, CH_MASK, :height, :] > 0.5
        paired = torch.stack((vs, zeta), dim=-1).permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[-1], -1)
        obs = observed.permute(0, 2, 1).repeat_interleave(2, dim=-1)
        filled = torch.where(obs, paired, self.mean)
        coded = (filled - self.mean) * self.weight
        code = coded @ self.components.transpose(0, 1)
        return code.transpose(1, 2).unsqueeze(2)


class FrozenColumnPCA(nn.Module):
    """Project each soil column onto a frozen masked-PCA basis.

    Features are log(Vs/400) and log(zeta/0.02), interleaved from the surface.
    Rows the column does not have stay at the mean, so they add nothing.
    The basis is a buffer: it is not a parameter and it does not train.
    """

    def __init__(self, mean: np.ndarray, components: np.ndarray):
        super().__init__()
        mean_t = torch.as_tensor(np.asarray(mean, dtype=np.float32))
        comp_t = torch.as_tensor(np.asarray(components, dtype=np.float32))
        if comp_t.ndim != 2 or mean_t.shape != (comp_t.shape[1],):
            raise ValueError(f"PCA mean {tuple(mean_t.shape)} does not match components {tuple(comp_t.shape)}")
        if comp_t.shape[1] % 2:
            raise ValueError("PCA features must be interleaved Vs and zeta")
        self.register_buffer("mean", mean_t)
        self.register_buffer("components", comp_t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) canvas -> (B, rank, 1, W) codes."""
        height = int(self.components.shape[1] // 2)
        vs = x[:, 0, :height, :]
        zeta = x[:, 1, :height, :]
        observed = x[:, CH_MASK, :height, :] > 0.5
        paired = torch.stack((vs, zeta), dim=-1).permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[-1], -1)
        obs = observed.permute(0, 2, 1).repeat_interleave(2, dim=-1)
        filled = torch.where(obs, paired, self.mean)
        code = (filled - self.mean) @ self.components.transpose(0, 1)
        return code.transpose(1, 2).unsqueeze(2)


def load_vardomain_state(model: VarDomainLOGLO, state) -> None:
    """Load a checkpoint, keeping new buffers when an older file lacks them."""
    state = dict(state)
    own = model.state_dict()
    for key, tensor in own.items():
        if key not in state:
            state[key] = tensor.detach().cpu().clone()
    model.load_state_dict(state)


def model_from_config(
    cfg: dict,
    modes: np.ndarray,
    mean: np.ndarray,
    pca_mean: np.ndarray | None = None,
    pca_components: np.ndarray | None = None,
    input_weight: np.ndarray | None = None,
) -> VarDomainLOGLO:
    """Build the architecture a checkpoint config names, including the lateral cutoff."""
    hfp = int(cfg.get("hfp", 4))
    model = VarDomainLOGLO(
        cfg["encoding"], modes, mean,
        n_modes_x=cfg.get("modes_x"),
        narrow=int(cfg.get("narrow", 32)),
        latent=int(cfg.get("latent", 128)),
        n_layers=int(cfg.get("layers", 5)),
        pointwise=bool(cfg.get("pointwise", False)) and str(cfg.get("readout", "pod")) == "pod",
        hfp_kernel=int(cfg.get("hfp_kernel", hfp)),
        hfp_stride=int(cfg.get("hfp_stride", hfp)),
        local_min_wavelength=float(cfg.get("local_min_wavelength") or 0.0),
        readout=str(cfg.get("readout", "pod")),
        depth_ae=bool(cfg.get("depth_ae", False)),
        recon_weight=float(cfg.get("recon_weight", 0.1)),
        depth_pca=bool(cfg.get("depth_pca", False)),
        pca_mean=pca_mean,
        pca_components=pca_components,
        translation_equivariant=bool(cfg.get("translation_equivariant", False)),
        contact_features=bool(cfg.get("contact_features", False)),
        contact_mean=None if cfg.get("contact_mean") is None else np.asarray(cfg["contact_mean"], dtype=np.float32),
        contact_std=None if cfg.get("contact_std") is None else np.asarray(cfg["contact_std"], dtype=np.float32),
        input_weight=input_weight,
    )
    if cfg.get("deltaphi"):
        from deltaphi import DeltaPhiHead

        model.deltaphi = DeltaPhiHead(2 * int(cfg.get("latent", 128)), int(modes.shape[1]))
    return model


def reconstruct_pod(
    mean: torch.Tensor, coeff: torch.Tensor, modes: torch.Tensor, weight: torch.Tensor,
) -> torch.Tensor:
    """Undo a diagonal frequency weight after the coefficient–mode product."""
    return mean + torch.einsum("bkm,mf->bkf", coeff, modes) / weight


def coefficient_scale(
    curves: np.ndarray, modes: np.ndarray, mean: np.ndarray, weight: np.ndarray | None = None,
) -> np.ndarray:
    """Per-mode standard deviation of POD coefficients. The branch predicts the whitened value."""
    centered = np.asarray(curves, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    if weight is not None:
        centered = centered * np.asarray(weight, dtype=np.float64)
    coeff = centered @ np.asarray(modes, dtype=np.float64).T
    sigma = coeff.std(axis=0)
    return np.maximum(sigma, 1e-3).astype(np.float32)


def extend_pod_modes(
    modes: np.ndarray, mean: np.ndarray, curves: np.ndarray, n_extra: int, weight: np.ndarray | None = None,
) -> np.ndarray:
    """Append up to ``n_extra`` modes spanning what ``modes`` does not already represent.

    New rows are orthonormal and orthogonal to ``modes``. Curves already in the
    old span produce no new rows. ``mean`` is left unchanged. A frequency weight
    is applied before the projection, matching a basis from ``fit_weighted_pod``.
    """
    modes64 = np.asarray(modes, dtype=np.float64)
    centered = np.asarray(curves, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    if weight is not None:
        centered = centered * np.asarray(weight, dtype=np.float64)
    resid = centered - (centered @ modes64.T) @ modes64
    if np.linalg.norm(resid) < 1e-8 or n_extra <= 0:
        return np.asarray(modes, dtype=np.float32)
    extra, _ = fit_log_pod(resid, n_extra)
    extra = np.asarray(extra, dtype=np.float64)
    kept = []
    for i in range(extra.shape[0]):
        v = extra[i].copy()
        v = v - (v @ modes64.T) @ modes64
        if kept:
            prev = np.stack(kept)
            v = v - (v @ prev.T) @ prev
        nrm = float(np.linalg.norm(v))
        if nrm >= 1e-6:
            kept.append(v / nrm)
    if not kept:
        return np.asarray(modes, dtype=np.float32)
    return np.concatenate([modes64, np.stack(kept)], axis=0).astype(np.float32)


# Rank and frequency weight chosen on the seed-0 residual holdout. At 128 modes,
# w(f) = sqrt(f / 1 Hz) has the lowest 5–10 Hz error. The same weight multiplies
# the stage-0 log-L1. A flat weight is the old 64-mode basis.
POD_MODES = 128
F_REF_HZ = 1.0


def spectral_weight(freq: np.ndarray) -> np.ndarray:
    """sqrt(f / 1 Hz). It is 1 at 1 Hz, about 0.32 at 0.1 Hz, and about 3.16 at 10 Hz."""
    return np.sqrt(np.asarray(freq, dtype=np.float64) / F_REF_HZ)


def fit_log_pod(curves: np.ndarray, n_modes: int) -> tuple[np.ndarray, np.ndarray]:
    """(K, F) modes and (F,) mean of log |TF| curves (rows) by eigen-decomposition of the covariance."""
    c = np.asarray(curves, dtype=np.float64)
    mean = c.mean(axis=0)
    centered = c - mean
    cov = centered.T @ centered / max(c.shape[0] - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1][:n_modes]
    return vecs[:, order].T.astype(np.float32), mean.astype(np.float32)


def fit_weighted_pod(curves: np.ndarray, n_modes: int, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """POD of w(f) times the centered curves. Reconstruct with ``reconstruct_pod``.

    The returned modes are orthonormal in the weighted coordinates. Dividing the
    mode product by ``weight`` puts the curve back in log|TF| units.
    """
    w = np.asarray(weight, dtype=np.float64)
    if w.shape != (np.asarray(curves).shape[1],) or np.any(w <= 0):
        raise ValueError("weight must be a positive value at each frequency")
    c = np.asarray(curves, dtype=np.float64)
    mean = c.mean(axis=0)
    modes, _ = fit_log_pod((c - mean) * w, n_modes)
    return modes, mean.astype(np.float32)
