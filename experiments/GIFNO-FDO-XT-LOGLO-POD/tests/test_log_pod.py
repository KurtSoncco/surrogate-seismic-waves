"""Log-space POD basis and exp readout. Synthetic tensors only."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import torch

from pod_readout import RecorderPODDeepONetHeadXT

_POD_SCRIPT = (
    Path(__file__).resolve().parents[2] / "GIFNO" / "preprocess" / "compute_pod_basis.py"
)


def _pod():
    spec = importlib.util.spec_from_file_location("compute_pod_basis_log", _POD_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {_POD_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _synthetic():
    rng = np.random.default_rng(0)
    tf = rng.uniform(0.2, 30.0, size=(6, 2, 8)).astype(np.float32)
    train_idx = np.arange(4)
    recorder_x = np.arange(2)
    return tf, train_idx, recorder_x


def test_linear_space_matches_the_default_svd():
    pod = _pod()
    tf, train_idx, recorder_x = _synthetic()
    default_modes, default_mean = pod.compute_pod_modes(tf, train_idx, recorder_x, 2)
    linear_modes, linear_mean = pod.compute_pod_modes(
        tf, train_idx, recorder_x, 2, space="linear"
    )
    np.testing.assert_array_equal(default_modes, linear_modes)
    np.testing.assert_array_equal(default_mean, linear_mean)


def test_log_mean_is_the_mean_of_log_amplitude():
    pod = _pod()
    tf, train_idx, recorder_x = _synthetic()
    _modes, mean = pod.compute_pod_modes(
        tf, train_idx, recorder_x, 3, space="log"
    )
    expected = np.log(np.clip(tf[train_idx, 0], pod.LOG_TF_EPS, None)).mean(axis=0)
    np.testing.assert_allclose(mean[0], expected, rtol=1e-5, atol=1e-5)


def test_shared_log_pod_ignores_unlabeled_zeros():
    pod = _pod()
    rng = np.random.default_rng(0)
    tf = np.zeros((4, 5, 8), dtype=np.float32)
    tf[:, :2] = rng.uniform(0.5, 2.0, size=(4, 2, 8))
    valid = np.zeros((4, 5), dtype=bool)
    valid[:, :2] = True
    train_idx = np.arange(4)
    _, mean_valid = pod.compute_shared_pod_modes(
        tf, train_idx, 3, space="log", valid=valid
    )
    _, mean_fallback = pod.compute_shared_pod_modes(tf, train_idx, 3, space="log")
    expected = np.log(tf[:, :2].astype(np.float64)).mean(axis=(0, 1))
    np.testing.assert_allclose(mean_valid, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(mean_fallback, expected, rtol=1e-5, atol=1e-5)
    assert float(mean_valid.max()) > -1.0


def test_shared_pod_reconstructs_a_train_curve():
    pod = _pod()
    tf, train_idx, _recorder_x = _synthetic()
    modes, mean = pod.compute_shared_pod_modes(tf, train_idx, 3, space="linear")
    assert modes.shape == (3, tf.shape[-1])
    assert mean.shape == (tf.shape[-1],)
    curve = tf[0, 0].astype(np.float64)
    coeffs = (curve - mean) @ modes.T
    recon = mean + coeffs @ modes
    # 3 modes on 8 bins cannot be exact; residual must beat the mean-only fit.
    err = np.linalg.norm(recon - curve)
    err_mean = np.linalg.norm(curve - mean)
    assert err < err_mean


def test_log_reconstruction_is_positive():
    pod = _pod()
    tf, train_idx, recorder_x = _synthetic()
    modes, mean = pod.compute_pod_modes(tf, train_idx, recorder_x, 3, space="log")
    work = pod.curves_in_space(tf[0].astype(np.float64), "log")
    coeffs = np.einsum("rf,rkf->rk", work - mean, modes)
    recon = np.exp(mean + np.einsum("rk,rkf->rf", coeffs, modes))
    assert recon.min() > 0.0
    np.testing.assert_allclose(recon, tf[0], rtol=1e-4, atol=1e-4)


def _head(space: str, shared: bool = False) -> RecorderPODDeepONetHeadXT:
    if shared:
        modes = np.zeros((1, 4), dtype=np.float32)
        modes[0, 0] = 1.0
        mean = np.log(np.full(4, 2.0, dtype=np.float32))
    else:
        modes = np.zeros((2, 1, 4), dtype=np.float32)
        modes[:, 0, 0] = 1.0
        mean = np.log(
            np.array([[2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]], dtype=np.float32)
        )
    head = RecorderPODDeepONetHeadXT(
        branch_in_channels=4,
        nx=4,
        n_freq=4,
        recorder_x=np.array([0, 1]),
        pod_modes=modes,
        pod_mean=mean,
        branch_hidden=4,
        output_space=space,
    )
    last = head.branch[-1]
    torch.nn.init.zeros_(last.weight)
    torch.nn.init.zeros_(last.bias)
    return head


def test_log_recon_clamps_insane_coefficients():
    head = _head("log", shared=True)
    last = head.branch[-1]
    torch.nn.init.zeros_(last.weight)
    torch.nn.init.constant_(last.bias, 80.0)
    x = torch.zeros(1, 2, 1, 4)
    out = head(x, x)
    assert torch.isfinite(out).all()
    assert float(out.max().detach()) <= math.exp(head.log_clamp) + 1e-5


def test_exp_readout_matches_exp_of_the_mean_and_shares_the_state_dict():
    linear = _head("linear")
    logged = _head("log")
    assert set(linear.state_dict()) == set(logged.state_dict())
    x = torch.zeros(1, 2, 1, 4)
    linear_out = linear.predict_recorders(x, x).detach().numpy()
    log_out = logged.predict_recorders(x, x).detach().numpy()
    np.testing.assert_allclose(log_out, np.exp(linear_out), rtol=1e-5, atol=1e-5)
    assert log_out.min() > 0.0


def test_shared_head_predicts_all_columns_and_recorders_agree():
    head = _head("linear", shared=True)
    last = head.branch[-1]
    torch.nn.init.zeros_(last.weight)
    torch.nn.init.zeros_(last.bias)
    x = torch.zeros(1, 2, 1, 4)
    full = head(x, x).detach()
    rec = head.predict_recorders(x, x).detach()
    assert full.shape == (1, 4, 4)
    assert rec.shape == (1, 2, 4)
    torch.testing.assert_close(full[:, :2], rec)


def test_reinit_pod_branch_resets_only_the_coefficient_layer():
    from train import _reinit_pod_branch_last_linear

    head = _head("log")
    first = head.branch[0]
    last = head.branch[-1]
    with torch.no_grad():
        first.weight.fill_(2.0)
        last.weight.fill_(1.0)
        last.bias.fill_(1.0)

    class _Model:
        pass

    model = _Model()
    model.head = head
    _reinit_pod_branch_last_linear(model)
    assert torch.allclose(first.weight, torch.full_like(first.weight, 2.0))
    assert not torch.allclose(last.weight, torch.ones_like(last.weight))


def test_create_model_uses_pod_space_from_config():
    import model as model_mod

    model_mod.config.POD_SPACE = "log"
    model = model_mod.create_model(num_fno_layers=1)
    assert model.head.output_space == "log"
    model_mod.config.POD_SPACE = "linear"
    model = model_mod.create_model(num_fno_layers=1)
    assert model.head.output_space == "linear"
