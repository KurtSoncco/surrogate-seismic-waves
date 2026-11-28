"""Checkpoint identity and the forward that matches training.

Training selects the epoch under CUDA bfloat16 autocast when ``config["amp"]``
is true. ``load_model`` binds that same autocast, and every saved array records
the checkpoint path, the weight hash, and the dtype that actually ran.
A float32 forward of the same weights is a different function.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

STAMP_KEYS = ("checkpoint", "weight_sha256", "autocast", "inference_dtype")


def weight_sha256(state_dict) -> str:
    """SHA-256 of parameter and buffer bytes, including the POD basis stored in the checkpoint."""
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().to(device="cpu").contiguous()
        digest.update(key.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        payload = torch.view_as_real(tensor) if tensor.is_complex() else tensor
        digest.update(payload.numpy().tobytes())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_stamp(model_dir: Path) -> dict:
    """Path, epoch, weight hash, and the POD files sitting beside the checkpoint."""
    model_dir = Path(model_dir).resolve()
    blob = torch.load(model_dir / "best_model.pt", map_location="cpu", weights_only=False)
    config = blob.get("config") or {}
    epoch = blob.get("epoch")
    return {
        "checkpoint": str(model_dir),
        "epoch": None if epoch is None else int(epoch),
        "weight_sha256": weight_sha256(blob["state_dict"]),
        "pod_modes_sha256": file_sha256(model_dir / "pod_modes.npy"),
        "pod_mean_sha256": file_sha256(model_dir / "pod_mean.npy"),
        "target": config.get("target"),
        "amp": bool(config.get("amp", True)),
        "stage": config.get("stage"),
        "seed": config.get("seed"),
    }


def inference_fields(device, amp: bool) -> dict:
    """Dtype ``load_model`` will use on this device."""
    autocast = bool(amp) and getattr(device, "type", None) == "cuda"
    return {
        "autocast": autocast,
        "inference_dtype": "bfloat16" if autocast else "float32",
    }


def full_stamp(model_dir: Path, device, amp: bool) -> dict:
    return {**checkpoint_stamp(model_dir), **inference_fields(device, amp)}


def audit_stamp_matches(saved: dict, expected: dict) -> bool:
    """Checkpoint identity plus the preregistration, data manifest, and source tree."""
    keys = STAMP_KEYS + ("preregistration_sha256", "manifest_sha256", "source_tree_sha256")
    return stamp_matches(saved, expected) and all(saved.get(key) == expected.get(key) for key in keys)


def stamp_matches(saved: dict, expected: dict) -> bool:
    """True when a sidecar names this checkpoint and this dtype, and is still in use."""
    if saved.get("discarded"):
        return False
    return all(saved.get(key) == expected.get(key) for key in STAMP_KEYS)


def bind_training_autocast(model, device, amp: bool):
    """Run ``model.forward`` under the autocast training used, when this device supports it."""
    if not inference_fields(device, amp)["autocast"]:
        return model
    raw = model.forward

    def forward(x, query, n_keep=None):
        enabled = x.device.type == "cuda"
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=enabled):
            return raw(x, query, n_keep=n_keep)

    model.forward = forward
    return model


def read_sidecar(path: Path) -> dict | None:
    side = Path(path).with_suffix(".json")
    if not side.is_file():
        return None
    return json.loads(side.read_text())


def write_sidecar(path: Path, stamp: dict) -> None:
    Path(path).with_suffix(".json").write_text(json.dumps(stamp, indent=2) + "\n")


def cache_is_current(path: Path, stamp: dict, n_rows: int) -> bool:
    """An array is reusable only when its sidecar matches and it has the expected row count."""
    path = Path(path)
    if not path.is_file():
        return False
    saved = read_sidecar(path)
    if saved is None or not stamp_matches(saved, stamp):
        return False
    import numpy as np

    array = np.load(path, mmap_mode="r")
    return int(array.shape[0]) == int(n_rows)
