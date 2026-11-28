#!/usr/bin/env python3
"""Where float32 and bfloat16 forwards of one checkpoint diverge.

Run 3848 is the record whose |TF| at 2.5 Hz is 2.73 in float32 and 3.99 in
bfloat16. Hooks and an explicit encoder trace locate the first layer that
moves. A float64 forward says whether float32 is the map of these weights.
The same output comparison is repeated for stage 0, stage 1, and the direct
model, and for the three-layer example.

    python precision_diagnose.py
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from data import KIND_RUN, Cache, make_window  # noqa: E402
from model import VarDomainLOGLO, load_vardomain_state, model_from_config, sample_grid  # noqa: E402

OUT = HERE / "precision"
CHECKPOINTS = {
    "stage2": REPO / "checkpoints" / "vardomain" / "early_width" / "stage2_rise50",
    "stage1": REPO / "checkpoints" / "vardomain" / "early_width" / "stage1",
    "stage0_residual": REPO / "checkpoints" / "vardomain" / "stage0_residual_s0",
    "stage0_direct": REPO / "checkpoints" / "vardomain" / "stage0_direct_s0",
}


def load_raw(model_dir: Path, device) -> tuple[VarDomainLOGLO, dict]:
    blob = torch.load(model_dir / "best_model.pt", map_location="cpu", weights_only=False)
    cfg = blob["config"]
    pca_mean = pca_components = None
    if cfg.get("depth_pca"):
        pca_mean = np.load(model_dir / "pca_mean.npy")
        pca_components = np.load(model_dir / "pca_components.npy")
    model = model_from_config(
        cfg, np.load(model_dir / "pod_modes.npy"), np.load(model_dir / "pod_mean.npy"), pca_mean, pca_components,
    )
    load_vardomain_state(model, blob["state_dict"])
    model.to(device).eval()
    return model, cfg


def window(cache: Cache, family: str, run: int):
    camp = next(c for c in cache.campaigns if c.name == family)
    base = next(i for i, (ci, _j) in enumerate(cache.run_of) if cache.campaigns[ci] is camp)
    x, query, _tf, prior, valid, info, _xa = make_window(
        cache, "metric", KIND_RUN, int(base + run), 500, 500, 0, False
    )
    mid = int(np.flatnonzero(valid.numpy())[valid.numpy().sum() // 2])
    return x.unsqueeze(0), query.unsqueeze(0), prior.numpy()[mid].astype(np.float64), mid, int(info[1])


def _zone(enabled: bool):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled)


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    den = torch.linalg.vector_norm(a).clamp_min(1e-12)
    return float(torch.linalg.vector_norm(a - b) / den)


def stem_encoder_head(model, x, query, *, stem: bool, encoder: bool, head: bool, n_keep: int | None = None):
    from encoding import CH_MASK

    with torch.no_grad():
        with _zone(stem):
            h = model.lift(x.index_select(1, model.lift_idx))
            h = model.collapse(h, x[:, CH_MASK])
            h = model.film(model.expand(h), model.site_groups(x))
        xg, xl = encoder_call(model, h, encoder)
        if not head:
            dtype = _real_dtype(model.branch)
            xg = xg.to(dtype=dtype)
            xl = xl.to(dtype=dtype)
        with _zone(head):
            grid = torch.cat([xg[:, :, 0, :], xl[:, :, 0, :]], dim=1)
            head_dtype = _real_dtype(model.branch)
            coeff_grid = model.branch(grid.transpose(1, 2).to(dtype=head_dtype)).transpose(1, 2)
            if query.ndim == 1:
                query = query.unsqueeze(0)
            coeff = sample_grid(coeff_grid, query).to(dtype=head_dtype)
            if n_keep is not None:
                coeff = coeff.clone()
                coeff[..., int(n_keep):] = 0
            log_tf = model.pod_mean + torch.einsum("bkm,mf->bkf", coeff, model.pod_modes)
            if model.pointwise is not None:
                log_tf = log_tf + model.pointwise(sample_grid(grid, query).to(dtype=head_dtype))
            log_tf = log_tf.clamp(-model.log_clamp, model.log_clamp)
    return log_tf, coeff, h


def _real_dtype(module) -> torch.dtype:
    for param in module.parameters():
        if param.is_floating_point() and not param.is_complex():
            return param.dtype
    return torch.float32


def encoder_call(model, h, enabled: bool):
    enc = model.encoder
    dtype = _real_dtype(enc)
    if not enabled:
        h = h.to(dtype=dtype)
    with _zone(enabled):
        x_global = h
        x_local = h
        for i in range(enc.n_layers):
            z = x_global + x_local
            x_hf = enc.hfp(z)
            with torch.autocast(device_type=h.device.type, enabled=False):
                x_g = enc.fno(x_global.to(dtype=dtype), index=i)
            x_l = enc.local_layers[i](x_local)
            hf_feat = enc.hf_mlps[i](x_hf)
            mix = x_g + x_l + hf_feat + enc.skips[i](z)
            gate = torch.sigmoid(enc.gates[i](mix))
            x_global = F.gelu(x_g + gate * (mix - x_g))
            x_local = F.gelu(x_l + (1.0 - gate) * (mix - x_l))
    return x_global, x_local


def encoder_layers(model, h, enabled: bool) -> list[dict[str, torch.Tensor]]:
    """One record of each tensor inside the encoder, under one autocast setting."""
    enc = model.encoder
    rows = []
    with torch.no_grad(), _zone(enabled):
        x_global = h
        x_local = h
        for i in range(enc.n_layers):
            z = x_global + x_local
            x_hf = enc.hfp(z)
            with torch.autocast(device_type=h.device.type, enabled=False):
                x_g = enc.fno(x_global.float(), index=i)
            x_l = enc.local_layers[i](x_local)
            hf_feat = enc.hf_mlps[i](x_hf)
            mix = x_g + x_l + hf_feat + enc.skips[i](z)
            gate = torch.sigmoid(enc.gates[i](mix))
            x_global = F.gelu(x_g + gate * (mix - x_g))
            x_local = F.gelu(x_l + (1.0 - gate) * (mix - x_l))
            rows.append(
                {
                    "fno": x_g.detach(),
                    "local": x_l.detach(),
                    "hf": hf_feat.detach(),
                    "gate": gate.detach(),
                    "x_global": x_global.detach(),
                    "x_local": x_local.detach(),
                }
            )
    return rows


def pod_cancellation(coeff: torch.Tensor, modes: torch.Tensor, mid: int) -> dict[str, float]:
    parts = coeff[0, mid].float().cpu()[:, None] * modes.float().cpu()
    net = parts.sum(0)
    gross = parts.abs().sum(0)
    ratio = gross / net.abs().clamp_min(1e-8)
    return {
        "mean_abs_net": float(net.abs().mean()),
        "median_gross_over_net": float(ratio.median()),
        "p90_gross_over_net": float(torch.quantile(ratio, 0.9)),
    }


def g_stats(residual: np.ndarray, prior: np.ndarray, freq: np.ndarray) -> dict[str, float]:
    """``residual`` is the network output g. |TF| is the column times e^g."""
    g = np.asarray(residual, dtype=np.float64)
    mid = (freq >= 0.4) & (freq <= 2.5)
    high = (freq >= 2.5) & (freq <= 10.0)

    def wiggle(band: np.ndarray) -> float:
        series = g[band]
        if series.size < 3:
            return 0.0
        return float(np.mean(np.abs(np.diff(series, n=2))))

    k = int(np.argmin(np.abs(freq - 2.5)))
    return {
        "tf_at_2p5": float(np.exp(prior[k] + g[k])),
        "g_at_2p5": float(g[k]),
        "mean_abs_g_mid": float(np.mean(np.abs(g[mid]))),
        "max_abs_g_mid": float(np.max(np.abs(g[mid]))),
        "mean_abs_g_high": float(np.mean(np.abs(g[high]))),
        "wiggle_mid": wiggle(mid),
        "wiggle_high": wiggle(high),
        "mean_abs_g": float(np.mean(np.abs(g))),
    }


def module_hooks(model, x, query, enabled: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    captured: dict[str, torch.Tensor] = {}
    hooks = []

    def make(name):
        def hook(_mod, _inp, out):
            tensor = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(tensor):
                captured[name] = tensor.detach()
        return hook

    for name, mod in model.named_modules():
        if name:
            hooks.append(mod.register_forward_hook(make(name)))
    with torch.no_grad(), _zone(enabled):
        out = model(x, query)
    for hook in hooks:
        hook.remove()
    return out, captured


def compare_modules(fp32: dict[str, torch.Tensor], bf16: dict[str, torch.Tensor]) -> list[dict]:
    rows = []
    for name in fp32:
        if name not in bf16 or fp32[name].shape != bf16[name].shape:
            continue
        rows.append({"module": name, "rel": rel(fp32[name], bf16[name]), "numel": int(fp32[name].numel())})
    rows.sort(key=lambda row: row["rel"], reverse=True)
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    cache = Cache(REPO / "data" / "vardomain" / "cache")
    freq = np.asarray(cache.freq, dtype=np.float64)
    x, query, prior, mid, _index = window(cache, "iid", 3848)
    x = x.to(device)
    query = query.to(device)
    report: dict = {"run": 3848, "family": "iid", "station": mid}

    model, cfg = load_raw(CHECKPOINTS["stage2"], device)
    report["stage2_config"] = {"target": cfg.get("target"), "amp": cfg.get("amp"), "epoch_modes": int(model.pod_modes.shape[0])}

    log_fp32, coeff_fp32, h_fp32 = stem_encoder_head(model, x, query, stem=False, encoder=False, head=False)
    log_bf16, coeff_bf16, h_bf16 = stem_encoder_head(model, x, query, stem=True, encoder=True, head=True)
    report["run_3848"] = {
        "float32": g_stats(log_fp32[0, mid].float().cpu().numpy(), prior, freq),
        "bfloat16": g_stats(log_bf16[0, mid].float().cpu().numpy(), prior, freq),
    }
    report["run_3848"]["rel_g"] = rel(log_fp32[0, mid] - torch.as_tensor(prior, device=device), log_bf16[0, mid] - torch.as_tensor(prior, device=device))
    report["run_3848"]["rel_output"] = rel(log_fp32, log_bf16)
    report["run_3848"]["rel_coeff"] = rel(coeff_fp32, coeff_bf16)
    report["run_3848"]["rel_encoder_input"] = rel(h_fp32, h_bf16)
    report["cancellation_float32"] = pod_cancellation(coeff_fp32, model.pod_modes, mid)
    report["cancellation_bfloat16"] = pod_cancellation(coeff_bf16, model.pod_modes, mid)

    # Extra modes only. Stage 2 grew 64 -> 80.
    log_fp32_64, coeff_fp32_64, _h = stem_encoder_head(model, x, query, stem=False, encoder=False, head=False, n_keep=64)
    log_bf16_64, coeff_bf16_64, _h = stem_encoder_head(model, x, query, stem=True, encoder=True, head=True, n_keep=64)
    report["n_keep_64"] = {
        "float32": g_stats(log_fp32_64[0, mid].float().cpu().numpy(), prior, freq),
        "bfloat16": g_stats(log_bf16_64[0, mid].float().cpu().numpy(), prior, freq),
        "rel_coeff_first64": rel(coeff_fp32[..., :64], coeff_bf16[..., :64]),
        "rel_coeff_extra": rel(coeff_fp32[..., 64:], coeff_bf16[..., 64:]),
        "mean_abs_extra_coeff_float32": float(coeff_fp32[..., 64:].abs().mean()),
        "mean_abs_extra_coeff_bfloat16": float(coeff_bf16[..., 64:].float().abs().mean()),
        "mean_abs_first64_coeff_float32": float(coeff_fp32[..., :64].abs().mean()),
        "mean_abs_first64_coeff_bfloat16": float(coeff_bf16[..., :64].float().abs().mean()),
    }

    # Which block moves the output when the others stay in float32.
    blocks = {}
    for name, flags in {
        "stem_only": (True, False, False),
        "encoder_only": (False, True, False),
        "head_only": (False, False, True),
        "encoder_and_head": (False, True, True),
    }.items():
        log_i, _c, _h = stem_encoder_head(model, x, query, stem=flags[0], encoder=flags[1], head=flags[2])
        blocks[name] = {
            "tf_at_2p5": g_stats(log_i[0, mid].float().cpu().numpy(), prior, freq)["tf_at_2p5"],
            "rel_to_float32": rel(log_fp32, log_i),
            "rel_to_bfloat16": rel(log_bf16, log_i),
        }
    report["block_swap"] = blocks

    # Head split: branch matmul versus the POD einsum, on the float32 encoder state.
    with torch.no_grad():
        h = h_fp32
        xg, xl = encoder_call(model, h, False)
        grid = torch.cat([xg[:, :, 0, :], xl[:, :, 0, :]], dim=1)
        with _zone(True):
            coeff_b = sample_grid(model.branch(grid.transpose(1, 2).float()).transpose(1, 2), query).float()
        with _zone(False):
            coeff_f = sample_grid(model.branch(grid.transpose(1, 2).float()).transpose(1, 2), query).float()
        with _zone(True):
            y_einsum_bf = model.pod_mean + torch.einsum("bkm,mf->bkf", coeff_f, model.pod_modes)
        with _zone(False):
            y_einsum_fp = model.pod_mean + torch.einsum("bkm,mf->bkf", coeff_b, model.pod_modes)
    report["head_split_on_float32_encoder"] = {
        "branch_rel": rel(coeff_f, coeff_b),
        "einsum_of_float32_coeff_in_bfloat16_rel": rel(log_fp32, y_einsum_bf),
        "einsum_of_bfloat16_coeff_in_float32_rel": rel(log_fp32, y_einsum_fp),
        "tf_einsum_bf16": g_stats(y_einsum_bf[0, mid].float().cpu().numpy(), prior, freq)["tf_at_2p5"],
        "tf_branch_bf16_einsum_fp32": g_stats(y_einsum_fp[0, mid].float().cpu().numpy(), prior, freq)["tf_at_2p5"],
    }
    lin1, gelu, lin2 = model.branch[0], model.branch[1], model.branch[2]
    hidden_in = grid.transpose(1, 2).float()
    with torch.no_grad():
        with _zone(False):
            h1_f = gelu(lin1(hidden_in))
        with _zone(True):
            h1_b = gelu(lin1(hidden_in))
        with _zone(False):
            c_second_fp = lin2(h1_b.float())
        with _zone(True):
            c_first_fp = lin2(h1_f.float())
    report["branch_layers_on_float32_features"] = {
        "first_linear_rel": rel(h1_f, h1_b),
        "second_linear_on_bf16_hidden_rel": rel(lin2(h1_f.float()), c_second_fp),
        "tf_first_bf16_second_fp32": g_stats(
            (model.pod_mean + torch.einsum("bwm,mf->bwf", sample_grid(c_second_fp.transpose(1, 2), query).float(), model.pod_modes))[0, mid].float().cpu().numpy(),
            prior,
            freq,
        )["tf_at_2p5"],
        "tf_first_fp32_second_bf16": g_stats(
            (model.pod_mean + torch.einsum("bwm,mf->bwf", sample_grid(c_first_fp.transpose(1, 2), query).float(), model.pod_modes))[0, mid].float().cpu().numpy(),
            prior,
            freq,
        )["tf_at_2p5"],
    }
    weight = lin2.weight.detach().float().cpu()
    column = int(query[0, mid].round().item())
    hidden = h1_f[0, column].detach().float().cpu()
    parts = hidden[:, None] * weight.T
    net = parts.sum(0)
    gross = parts.abs().sum(0)
    report["second_linear_cancellation"] = {
        "median_gross_over_net": float((gross / net.abs().clamp_min(1e-8)).median()),
        "max_abs_weight": float(weight.abs().max()),
        "median_abs_hidden": float(hidden.abs().median()),
    }

    # Encoder layers, fed the float32 stem so the trace starts from one input.
    layers_fp = encoder_layers(model, h_fp32, False)
    layers_bf = encoder_layers(model, h_fp32, True)
    layer_rows = []
    for i, (a, b) in enumerate(zip(layers_fp, layers_bf)):
        layer_rows.append({"layer": i, **{key: rel(a[key], b[key]) for key in a}})
    report["encoder_layers_same_input"] = layer_rows

    # Full-module hooks on the real forward.
    out_fp, hooks_fp = module_hooks(model, x, query, False)
    out_bf, hooks_bf = module_hooks(model, x, query, True)
    hooked = compare_modules(hooks_fp, hooks_bf)
    report["largest_module_rel"] = hooked[:25]
    report["hook_output_agrees_with_split"] = {
        "rel_hook_outputs": rel(out_fp, out_bf),
        "rel_split_outputs": rel(log_fp32, log_bf16),
    }

    # float64 of the same weights. The shipped forward hard-casts the Fourier
    # layers to float32, so this path keeps each layer in the weight dtype.
    from spectral_layers import LocalPatchSpectralConv2d

    model64 = copy.deepcopy(model).double()
    for module in model64.modules():
        weight = getattr(module, "weight", None)
        tensor = getattr(weight, "tensor", None)
        if torch.is_tensor(tensor) and tensor.dtype == torch.complex64:
            tensor.data = tensor.data.to(torch.complex128)
    original_local = LocalPatchSpectralConv2d.forward
    LocalPatchSpectralConv2d.forward = _local_fft_in_weight_dtype
    try:
        y64, _c64, _h64 = stem_encoder_head(model64, x.double(), query.double(), stem=False, encoder=False, head=False)
    finally:
        LocalPatchSpectralConv2d.forward = original_local
    y64 = y64.float()
    report["float64"] = g_stats(y64[0, mid].cpu().numpy(), prior, freq)
    report["float64"]["note"] = "Fourier casts to float32 were removed so the whole forward stays float64."
    report["float64"]["rel_to_float32"] = rel(log_fp32, y64)
    report["float64"]["rel_to_bfloat16"] = rel(log_bf16, y64)
    del model64
    torch.cuda.empty_cache()

    # Other checkpoints, same record, output only.
    others = {}
    for name, path in CHECKPOINTS.items():
        if not (path / "best_model.pt").is_file():
            others[name] = {"missing": str(path / "best_model.pt")}
            continue
        other, other_cfg = load_raw(path, device)
        y0, _c0, _h0 = stem_encoder_head(other, x, query, stem=False, encoder=False, head=False)
        y1, _c1, _h1 = stem_encoder_head(other, x, query, stem=True, encoder=True, head=True)
        others[name] = {
            "target": other_cfg.get("target"),
            "n_modes": int(other.pod_modes.shape[0]),
            "float32_tf_at_2p5": g_stats(y0[0, mid].float().cpu().numpy(), prior, freq)["tf_at_2p5"],
            "bfloat16_tf_at_2p5": g_stats(y1[0, mid].float().cpu().numpy(), prior, freq)["tf_at_2p5"],
            "rel_output": rel(y0, y1),
            "mean_abs_log_diff": float((y0 - y1).float().abs().mean()),
        }
        del other
        torch.cuda.empty_cache()
    report["checkpoints_on_run_3848"] = others

    # A few more horizontal runs, stage 2 only, so 3848 is not the whole claim.
    split = np.load(REPO / "data" / "vardomain" / "cache" / "iid" / "split.npy")
    runs = np.flatnonzero(np.asarray(split).astype(str) == "test")
    pick = [3848] + [int(r) for r in runs[np.linspace(0, runs.size - 1, 7).round().astype(int)] if int(r) != 3848]
    stage2, _cfg = load_raw(CHECKPOINTS["stage2"], device)
    multi = []
    for run in pick:
        xx, qq, pr, md, _i = window(cache, "iid", int(run))
        y0, _c, _h = stem_encoder_head(stage2, xx.to(device), qq.to(device), stem=False, encoder=False, head=False)
        y1, _c, _h = stem_encoder_head(stage2, xx.to(device), qq.to(device), stem=True, encoder=True, head=True)
        s0 = g_stats(y0[0, md].float().cpu().numpy(), pr, freq)
        s1 = g_stats(y1[0, md].float().cpu().numpy(), pr, freq)
        multi.append(
            {
                "run": int(run),
                "tf_float32": s0["tf_at_2p5"],
                "tf_bfloat16": s1["tf_at_2p5"],
                "mean_abs_log_diff": float((y0[0, md] - y1[0, md]).float().abs().mean()),
                "rel_output": rel(y0, y1),
            }
        )
    report["stage2_more_runs"] = multi
    report["stage2_more_runs_mean_abs_log_diff"] = float(np.mean([row["mean_abs_log_diff"] for row in multi]))

    # Three-layer example. The saved g is the bfloat16 dump.
    example = np.load(HERE / "paper" / "dump" / "three_layer_example.npz", allow_pickle=True)
    record = str(np.asarray(example["record_id"]).reshape(-1)[0])
    family, local = record.split(":")
    xx, qq, pr, md, _i = window(cache, family, int(local))
    y0, _c, _h = stem_encoder_head(stage2, xx.to(device), qq.to(device), stem=False, encoder=False, head=False)
    y1, _c, _h = stem_encoder_head(stage2, xx.to(device), qq.to(device), stem=True, encoder=True, head=True)
    saved_g = np.asarray(example["g_stage2"], dtype=np.float64)
    g0 = y0[0, md].float().cpu().numpy()
    g1 = y1[0, md].float().cpu().numpy()
    report["three_layer"] = {
        "record_id": record,
        "float32": g_stats(y0[0, md].float().cpu().numpy(), pr, freq),
        "bfloat16": g_stats(y1[0, md].float().cpu().numpy(), pr, freq),
        "saved_dump_wiggle_mid": float(np.mean(np.abs(np.diff(saved_g[(freq >= 0.4) & (freq <= 2.5)], n=2)))),
        "saved_dump_max_abs_g_mid": float(np.max(np.abs(saved_g[(freq >= 0.4) & (freq <= 2.5)]))),
        "corr_float32_vs_saved": _corr(g0, saved_g),
        "corr_bfloat16_vs_saved": _corr(g1, saved_g),
        "mean_abs_g_diff_float32_vs_bfloat16": float(np.mean(np.abs(g0 - g1))),
    }

    (OUT / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    _plot(
        layer_rows,
        report,
        log_fp32[0, mid].float().cpu().numpy(),
        log_bf16[0, mid].float().cpu().numpy(),
        y64[0, mid].cpu().numpy(),
        prior,
        freq,
    )
    print(json.dumps({k: report[k] for k in ("run_3848", "n_keep_64", "block_swap", "head_split_on_float32_encoder", "float64", "checkpoints_on_run_3848", "three_layer", "encoder_layers_same_input")}, indent=2))
    print(f"wrote {OUT}")


def _local_fft_in_weight_dtype(self, x: torch.Tensor) -> torch.Tensor:
    """Local spectral conv without the hard float32 FFT cast."""
    b, c, h, w = x.shape
    ph, pw = self.ph, self.pw
    n_h, n_w = h // ph, w // pw
    patches = (
        x.reshape(b, c, n_h, ph, n_w, pw).permute(0, 2, 4, 1, 3, 5).reshape(b * n_h * n_w, c, ph, pw)
    )
    fft_dtype = self.weight.real.dtype
    x_ft = torch.fft.rfft2(patches.to(dtype=fft_dtype), norm="ortho")
    out = torch.fft.irfft2(torch.einsum("bixy,ijxy->bjxy", x_ft, self.weight), s=(ph, pw), norm="ortho")
    return (
        out.reshape(b, n_h, n_w, c, ph, pw).permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w).to(dtype=x.dtype)
    )


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / max(np.sqrt((a * a).sum() * (b * b).sum()), 1e-12))


def _plot(layers, report, log_fp32, log_bf16, log_fp64, prior, freq) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    names = ["fno", "local", "hf", "gate", "x_global", "x_local"]
    for name in names:
        axes[0].plot([row["layer"] for row in layers], [row[name] for row in layers], marker="o", ms=3.5, lw=1.2, label=name)
    axes[0].set_xlabel("Encoder layer")
    axes[0].set_ylabel(r"$\|b-a\|/\|a\|$  (same stem input)")
    axes[0].set_title("Encoder, float32 input")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    axes[0].set_xticks(range(len(layers)))

    base = report["run_3848"]["float32"]["tf_at_2p5"]
    labels = ["stem", "encoder", "first branch linear", "POD einsum", "last branch linear"]
    vals = [
        report["block_swap"]["stem_only"]["tf_at_2p5"] - base,
        report["block_swap"]["encoder_only"]["tf_at_2p5"] - base,
        report["branch_layers_on_float32_features"]["tf_first_bf16_second_fp32"] - base,
        report["head_split_on_float32_encoder"]["tf_einsum_bf16"] - base,
        report["branch_layers_on_float32_features"]["tf_first_fp32_second_bf16"] - base,
    ]
    colors = ["#D55E00" if abs(v) > 0.2 else "#4C72B0" for v in vals]
    axes[1].barh(labels[::-1], vals[::-1], color=colors[::-1])
    axes[1].axvline(0, color="0.4", lw=0.6)
    axes[1].set_xlabel(r"$\Delta$|TF| at 2.5 Hz vs float32")
    axes[1].set_title("One block in bfloat16")
    fig.tight_layout()
    fig.savefig(OUT / "layer_rel.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.semilogx(freq, np.exp(prior + log_fp64), color="0.35", lw=1.4, label="float64")
    ax.semilogx(freq, np.exp(prior + log_fp32), color="#0072B2", lw=1.3, label="float32")
    ax.semilogx(freq, np.exp(prior + log_bf16), color="#D55E00", lw=1.3, label="bfloat16")
    ax.semilogx(freq, np.exp(prior), color="0.55", lw=1.0, ls="--", label="column")
    ax.set_xlim(0.1, 10)
    ax.set_xlabel("f (Hz)")
    ax.set_ylabel("|TF|")
    ax.set_title("Run 3848, center station")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "run3848_tf.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
