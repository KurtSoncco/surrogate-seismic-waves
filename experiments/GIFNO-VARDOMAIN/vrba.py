"""vRBA Algorithm 3 for operator learning: residual weights, family sampling, grad balance.

The inner weight is an EMA of Phi(r) over frequency. Quadratic Phi is r^2.
Exponential Phi is exp(r / eps) with eps annealed toward a floor. The outer
sample is the unified quota sampler. Gradient-norm balancing rescales each
loss so the parameter gradients share one mean norm (Eqs. 27–28).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VRBAWeights(nn.Module):
    def __init__(self, n_freq: int, ema: float = 0.9, potential: str = "quadratic", eps: float = 1.0):
        super().__init__()
        if potential not in ("quadratic", "exponential"):
            raise ValueError(f"potential must be quadratic or exponential, got {potential!r}")
        self.ema = float(ema)
        self.potential = potential
        self.eps0 = float(eps)
        self.register_buffer("freq_level", torch.ones(n_freq))
        self.register_buffer("seen", torch.zeros(()))

    def phi(self, residual: torch.Tensor, epoch: int) -> torch.Tensor:
        r = residual.abs()
        if self.potential == "quadratic":
            return r * r
        eps = max(self.eps0 * (0.5 ** (epoch / 20.0)), 0.05)
        return torch.exp((r / eps).clamp(max=20.0))

    def forward(self, residual: torch.Tensor, valid: torch.Tensor, epoch: int = 0) -> torch.Tensor:
        """(B, K, F) weights. The EMA is updated from the detached residual."""
        phi = self.phi(residual.detach(), epoch)
        mask = valid.to(phi.dtype)[:, :, None]
        count = mask.sum(dim=(0, 1)).clamp_min(1.0)
        mean = (phi * mask).sum(dim=(0, 1)) / count
        rate = self.ema if float(self.seen) > 0 else 0.0
        self.freq_level.mul_(rate).add_(mean * (1.0 - rate))
        self.seen += 1
        lam = phi / self.freq_level.clamp_min(1e-6)
        lam = lam * mask
        scale = lam.sum() / mask.sum().clamp_min(1.0)
        return lam / scale.clamp_min(1e-6)


def weighted_residual(residual: torch.Tensor, valid: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    mask = valid.to(residual.dtype)[:, :, None]
    err = residual.abs() * lam * mask
    return err.sum() / (lam * mask).sum().clamp_min(1e-6)


def grad_norm_balance(losses: dict[str, torch.Tensor], params: list[torch.nn.Parameter]) -> torch.Tensor:
    """Sum of losses scaled so each has the mean parameter-gradient norm.

    A loss with no gradient is left unscaled. The scales are detached.
    """
    if not losses:
        raise ValueError("grad_norm_balance needs at least one loss")
    live = [(name, loss) for name, loss in losses.items() if loss.requires_grad]
    if len(live) <= 1:
        return sum(losses.values())
    norms = []
    for _name, loss in live:
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        sq = loss.new_zeros(())
        for grad in grads:
            if grad is not None:
                sq = sq + grad.detach().float().pow(2).sum()
        norms.append(torch.sqrt(sq).clamp_min(1e-12))
    ref = torch.stack(norms).mean()
    scaled = []
    cursor = 0
    names = {name for name, _loss in live}
    for name, loss in losses.items():
        if name not in names:
            scaled.append(loss)
            continue
        scaled.append(loss * (ref / norms[cursor]).detach())
        cursor += 1
    return torch.stack([item.reshape(()) for item in scaled]).sum()
