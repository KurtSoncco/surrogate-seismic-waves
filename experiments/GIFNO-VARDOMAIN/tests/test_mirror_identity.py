"""The training mirror is an exact array flip. It is not an OpenSees pair.

``WindowSet`` flips half the time when ``allow_mirror`` is set. Vs and zeta
columns reverse, and a station at x is queried at width - x. The label stays
with the station. A 1D column on ``section_1d`` follows that column. Five
finite-element pairs were not solved.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

GD = Path(__file__).resolve().parents[1]
CASE = GD.parent / "case7a"
sys.path.insert(0, str(GD))
sys.path.insert(0, str(CASE))

from data import KIND_SECTION, Section, WindowSet, apply_mirror, make_window  # noqa: E402
from encoding import query_columns  # noqa: E402
from section_1d import surface_over_outcrop  # noqa: E402


def _section() -> Section:
    width = 40
    depth = 5
    column = np.arange(width, dtype=np.float64)
    vs = 160.0 + column[None, :] + np.arange(depth, dtype=np.float64)[:, None] * 30.0
    zeta = 0.02 + 0.001 * column[None, :] + np.arange(depth, dtype=np.float64)[:, None] * 0.002
    x_m = np.array([10.0, 20.0, 30.0])
    n_freq = 4
    log_tf = np.arange(x_m.size * n_freq, dtype=np.float32).reshape(x_m.size, n_freq)
    return Section(
        name="mirror-unit",
        vs=vs,
        zeta=zeta,
        x_m=x_m,
        log_tf=log_tf,
        log_tf1d=log_tf + 1.0,
        split="train",
        f1_hz=0.5,
        f2_hz=1.5,
    )


class _Cache:
    def __init__(self, section: Section):
        self.sections = [section]


def test_apply_mirror_reverses_columns_and_station_x_and_undoes():
    sec = _section()
    width = float(sec.lx)
    vs_m, ze_m, x_m = apply_mirror(sec.vs, sec.zeta, sec.x_m, width)
    np.testing.assert_array_equal(vs_m, sec.vs[:, ::-1])
    np.testing.assert_array_equal(ze_m, sec.zeta[:, ::-1])
    np.testing.assert_array_equal(x_m, width - sec.x_m)
    vs_2, ze_2, x_2 = apply_mirror(vs_m, ze_m, x_m, width)
    np.testing.assert_array_equal(vs_2, sec.vs)
    np.testing.assert_array_equal(ze_2, sec.zeta)
    np.testing.assert_array_equal(x_2, sec.x_m)


def test_make_window_flip_matches_the_reversed_arrays_and_keeps_the_label():
    sec = _section()
    cache = _Cache(sec)
    width = sec.lx
    plain = make_window(cache, "metric", KIND_SECTION, 0, width, 0, 0, False)
    mirrored = make_window(cache, "metric", KIND_SECTION, 0, width, 0, 0, True)
    from encoding import encode

    vs_m, ze_m, _x = apply_mirror(sec.vs, sec.zeta, sec.x_m, width)
    manual = encode("metric", vs_m, ze_m, sec.vs.shape[0], f1_hz=sec.f1_hz, f2_hz=sec.f2_hz)
    original = encode("metric", sec.vs, sec.zeta, sec.vs.shape[0], f1_hz=sec.f1_hz, f2_hz=sec.f2_hz)
    np.testing.assert_array_equal(plain[0].numpy(), original)
    np.testing.assert_array_equal(mirrored[0].numpy(), manual)
    assert not np.array_equal(plain[0].numpy(), mirrored[0].numpy())
    valid = plain[4].numpy()
    np.testing.assert_array_equal(plain[1].numpy()[valid], query_columns("metric", sec.x_m, width))
    np.testing.assert_array_equal(
        mirrored[1].numpy()[mirrored[4].numpy()],
        query_columns("metric", width - sec.x_m, width),
    )
    np.testing.assert_array_equal(plain[2].numpy(), mirrored[2].numpy())
    np.testing.assert_array_equal(mirrored[6].numpy()[mirrored[4].numpy()], width - sec.x_m)


def test_windowset_coin_is_the_half_flip_and_allow_mirror_can_forbid_it():
    sec = _section()
    cache = _Cache(sec)
    width = sec.lx
    plain = make_window(cache, "metric", KIND_SECTION, 0, width, 0, 0, False)
    mirrored = make_window(cache, "metric", KIND_SECTION, 0, width, 0, 0, True)
    record = {
        "kind": "section", "index": 0, "draw": 7, "fixed": True,
        "width": width, "start": 0, "stride": 0, "allow_mirror": True,
    }
    window = WindowSet(cache, "metric", [record], stage=0, seed=3, mirror=True)
    saw_flip = saw_plain = False
    for epoch in range(40):
        window.set_epoch(epoch)
        coin = np.random.default_rng([3, epoch, 7]).random() < 0.5
        got = window[0]
        ref = mirrored if coin else plain
        assert torch.equal(got[0], ref[0])
        assert torch.equal(got[1], ref[1])
        assert torch.equal(got[2], ref[2])
        saw_flip = saw_flip or coin
        saw_plain = saw_plain or not coin
        if saw_flip and saw_plain:
            break
    assert saw_flip and saw_plain
    record["allow_mirror"] = False
    held = WindowSet(cache, "metric", [record], stage=0, seed=3, mirror=True)
    for epoch in range(8):
        held.set_epoch(epoch)
        assert torch.equal(held[0][0], plain[0])
        assert torch.equal(held[0][1], plain[1])
    off = WindowSet(cache, "metric", [{**record, "allow_mirror": True}], stage=0, seed=3, mirror=False)
    for epoch in range(8):
        off.set_epoch(epoch)
        assert torch.equal(off[0][0], plain[0])


def test_section_1d_transfer_follows_the_mirrored_column():
    """One column pair. This is the linear SH operator, not a finite-element pair."""

    freq = np.geomspace(0.1, 10.0, 24)
    vs = np.array([
        [160.0, 420.0],
        [240.0, 680.0],
        [900.0, 1600.0],
    ])
    zeta = np.array([
        [0.02, 0.08],
        [0.03, 0.09],
        [0.005, 0.02],
    ])
    vs_m, ze_m, _x = apply_mirror(vs, zeta, np.array([0.0, 1.0]), 2.0)
    for index in range(vs.shape[1]):
        xi = np.repeat(zeta[:, index][:, None], freq.size, axis=1)
        xi_m = np.repeat(ze_m[:, -1 - index][:, None], freq.size, axis=1)
        original = surface_over_outcrop(freq, vs[:, index], dz=1.0, xi=xi)
        flipped = surface_over_outcrop(freq, vs_m[:, -1 - index], dz=1.0, xi=xi_m)
        np.testing.assert_allclose(flipped, original, rtol=1e-12, atol=1e-12)
    left = surface_over_outcrop(
        freq, vs[:, 0], dz=1.0, xi=np.repeat(zeta[:, 0][:, None], freq.size, axis=1),
    )
    right = surface_over_outcrop(
        freq, vs[:, 1], dz=1.0, xi=np.repeat(zeta[:, 1][:, None], freq.size, axis=1),
    )
    assert np.max(np.abs(left - right)) > 1e-3
