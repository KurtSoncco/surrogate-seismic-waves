# data_loader.py
"""Data loading for 2D Vs profile to transfer function prediction."""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


def _get_from_dict(d: Dict[str, Any], keys: Tuple[str, ...]) -> Any:
    """Return first value found for any of the given keys."""
    for k in keys:
        if k in d:
            return d[k]
    raise KeyError(f"None of {keys} found in dict keys: {list(d.keys())}")


def _pad_or_truncate_2d(
    arr: np.ndarray, target_shape: Tuple[int, int], mode: str = "edge"
) -> np.ndarray:
    """Pad or truncate 2D array to target (H, W)."""
    h, w = target_shape
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    arr = np.asarray(arr).squeeze()
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim > 2:
        arr = arr[0]
    ah, aw = arr.shape
    if ah < h or aw < w:
        padded = np.pad(arr, ((0, max(0, h - ah)), (0, max(0, w - aw))), mode=mode)
    else:
        padded = arr[:h, :w]
    return padded[:h, :w].astype(np.float32)


def _pad_or_truncate_1d(
    arr: np.ndarray, target_len: int, mode: str = "edge"
) -> np.ndarray:
    """Pad or truncate 1D array to target length."""
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    arr = np.asarray(arr).ravel()
    if len(arr) < target_len:
        padded = np.pad(arr, (0, target_len - len(arr)), mode=mode)
    else:
        padded = arr[:target_len]
    return padded.astype(np.float32)


class TTFDataset2D(Dataset):
    """PyTorch Dataset for 2D Vs profile to transfer function."""

    def __init__(
        self,
        vs_list: List[np.ndarray],
        tf_list: List[np.ndarray],
        vs_shape: Tuple[int, int],
        tf_len: int,
        vs_keys: Tuple[str, ...] = ("vs_2d", "vs", "Vs_profile", "Vs"),
        tf_keys: Tuple[str, ...] = ("tf", "transfer_function", "TTF", "ttf"),
    ):
        self.vs_shape = vs_shape
        self.tf_len = tf_len
        self.vs_keys = vs_keys
        self.tf_keys = tf_keys

        self.vs_tensors: List[torch.Tensor] = []
        self.tf_tensors: List[torch.Tensor] = []

        for vs_raw, tf_raw in zip(vs_list, tf_list):
            vs_arr = vs_raw if isinstance(vs_raw, np.ndarray) else np.array(vs_raw)
            tf_arr = tf_raw if isinstance(tf_raw, np.ndarray) else np.array(tf_raw)
            vs_proc = _pad_or_truncate_2d(vs_arr, vs_shape)
            tf_proc = _pad_or_truncate_1d(tf_arr, tf_len)
            self.vs_tensors.append(
                torch.tensor(vs_proc, dtype=torch.float32).unsqueeze(0)
            )
            self.tf_tensors.append(torch.tensor(tf_proc, dtype=torch.float32))

    def __len__(self) -> int:
        return len(self.tf_tensors)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.vs_tensors[idx], self.tf_tensors[idx]


def load_transfer_functions_dict(
    path: Path,
    vs_keys: Tuple[str, ...] = ("vs_array", "vs_2d", "vs", "Vs_profile", "Vs"),
    tf_keys: Tuple[str, ...] = (
        "tf_magnitude",
        "tf",
        "transfer_function",
        "TTF",
        "ttf",
    ),
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load dict {seed: {'vs_array': ..., 'tf_magnitude': ...}} and return (vs_list, tf_list).
    Handles wrapped format: {'dict_format': {seed: {...}}} or direct {seed: {...}}.
    Supports flexible key names via vs_keys and tf_keys.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    # Handle wrapped format: {'dict_format': {seed: {...}}}
    if isinstance(data, dict) and "dict_format" in data:
        data = data["dict_format"]

    vs_list: List[np.ndarray] = []
    tf_list: List[np.ndarray] = []

    for key, item in data.items():
        if not isinstance(item, dict):
            continue
        try:
            vs_raw = _get_from_dict(item, vs_keys)
            tf_raw = _get_from_dict(item, tf_keys)
        except KeyError:
            continue
        vs_arr = np.asarray(vs_raw, dtype=object)
        tf_arr = np.asarray(tf_raw, dtype=object)
        vs_list.append(vs_arr)
        tf_list.append(tf_arr)

    return vs_list, tf_list


def load_frequency_data(path: Path) -> np.ndarray:
    """
    Load frequency array from the transfer_functions_dict pickle.
    Uses the first sample's 'freq' from dict_format (all samples share the same grid).
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "dict_format" in data:
        data = data["dict_format"]
    first_key = next(iter(data))
    item = data[first_key]
    if not isinstance(item, dict) or "freq" not in item:
        raise KeyError("'freq' not found in dict_format entry")
    return np.asarray(item["freq"], dtype=np.float64)


def get_data_loaders(
    data_path: Optional[Path] = None,
    vs_shape: Tuple[int, int] = (64, 64),
    tf_len: int = 1000,
    vs_keys: Tuple[str, ...] = ("vs_2d", "vs", "Vs_profile", "Vs"),
    tf_keys: Tuple[str, ...] = ("tf", "transfer_function", "TTF", "ttf"),
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    batch_size: int = 64,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, and test DataLoaders from transfer_functions_dict.pkl."""
    import config as cfg

    path = data_path or cfg.DATA_PATH
    vs_list, tf_list = load_transfer_functions_dict(path, vs_keys, tf_keys)

    if not vs_list or not tf_list:
        raise ValueError(
            f"No valid samples loaded from {path}. "
            "Check vs_keys and tf_keys match your dict structure."
        )

    dataset = TTFDataset2D(vs_list, tf_list, vs_shape, tf_len, vs_keys, tf_keys)

    n = len(dataset)
    train_n = int(train_split * n)
    val_n = int(val_split * n)
    test_n = n - train_n - val_n

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_n, val_n, test_n], generator=gen
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    return train_loader, val_loader, test_loader
