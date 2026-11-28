"""Center-station |TF| already saved for the stage-2 test split."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from paths import CACHE, DUMP, FIG, REPO

FAMILIES = {
    "Horizontal": "iid",
    "Dipping": "ood_dipping",
}
H5 = {
    "iid": REPO / "data" / "gifno_screen" / "h5",
    "ood_dipping": REPO / "data" / "gifno_screen" / "ood_dipping" / "h5",
}
CENTER = 10


def test_frame(family: str) -> dict[str, np.ndarray]:
    """Align OpenSees, the column, the stage-2 model, and the saved classical curves.

    Rows follow ``split == test`` in the cache. Station 10 is the center recorder,
    the same index ``score_test_methods`` uses.
    """
    split = np.load(CACHE / family / "split.npy")
    runs = np.flatnonzero(np.asarray(split).astype(str) == "test").astype(np.int64)
    log_tf = np.asarray(np.load(CACHE / family / "log_tf.npy", mmap_mode="r")[runs, CENTER], dtype=np.float64)
    log_col = np.asarray(np.load(CACHE / family / "log_tf1d.npy", mmap_mode="r")[runs, CENTER], dtype=np.float64)
    group = np.load(CACHE / family / "group.npy")[runs]
    f1 = np.load(CACHE / family / "rayleigh_f1.npy")[runs]
    eval_tf = DUMP / f"ensemble_center_tf_{family}.npz"
    if not eval_tf.is_file():
        eval_tf = DUMP / f"stage2_center_tf_{family}.npz"
    if not eval_tf.is_file():
        raise FileNotFoundError(
            f"{eval_tf} is missing. Center-station tables read this eval-path npz. "
            f"The float32 copy is figures/stage2/discarded/test_model_{family}_stage2_epoch53_float32.npy."
        )
    blob = np.load(eval_tf)
    order = {int(run): i for i, run in enumerate(np.asarray(blob["run"]))}
    missing = [int(run) for run in runs if int(run) not in order]
    if missing:
        raise ValueError(f"{eval_tf} is missing runs {missing[:5]}")
    model = np.stack([np.asarray(blob["tf"][order[int(run)]], dtype=np.float64) for run in runs])
    arms_path = FIG / f"test_arms_{family}.npz"
    arms = np.load(arms_path)
    if not np.array_equal(np.asarray(arms["run"]), runs):
        raise ValueError(f"{arms_path} run ids do not match the test split")
    f0 = np.array([_f0(family, int(run), float(fallback)) for run, fallback in zip(runs, f1)], dtype=np.float64)
    freq = np.asarray(np.load(CACHE / "freq.npy"), dtype=np.float64)
    return {
        "run": runs,
        "group": np.asarray(group),
        "freq": freq,
        "f0": f0,
        "opensees": np.exp(log_tf),
        "column": np.exp(log_col),
        "model": model,
        "pretell": np.asarray(arms["pretell"], dtype=np.float64),
        "toro": np.asarray(arms["toro"], dtype=np.float64),
        "dmult": np.asarray(arms["dmult"], dtype=np.float64),
        "log_opensees": log_tf,
        "log_column": log_col,
        "log_model": np.log(np.clip(model, 1e-12, None)),
    }


def _f0(family: str, run: int, fallback: float) -> float:
    path = H5[family] / f"run_{run}.h5"
    if not path.is_file():
        return fallback
    import h5py

    with h5py.File(path, "r") as handle:
        attrs = handle["params"].attrs
        if "f0_effective" in attrs:
            return float(attrs["f0_effective"])
    return fallback


def passeri_path(family: str) -> Path:
    return FIG / f"test_passeri_{family}.npz"
