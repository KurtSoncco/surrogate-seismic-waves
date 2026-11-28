"""SHA-256 counter-mode seal and an append-only opening ledger.

The custodian holds the key. Ciphertext hashes can be published before training.
Decryption is refused until the ledger has an opening entry for that payload.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def sha256_ctr(key: bytes, nonce: bytes, data: bytes) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < len(data):
        block = hashlib.sha256(key + nonce + counter.to_bytes(8, "little")).digest()
        out.extend(block)
        counter += 1
    return bytes(a ^ b for a, b in zip(data, out[: len(data)]))


def seal_bytes(payload: bytes, key: bytes) -> dict:
    nonce = os.urandom(16)
    token = sha256_ctr(key, nonce, payload)
    return {
        "nonce_hex": nonce.hex(),
        "ciphertext_sha256": hashlib.sha256(token).hexdigest(),
        "ciphertext": token,
    }


def open_bytes(sealed: dict, key: bytes, *, ledger_opened: bool) -> bytes:
    if not ledger_opened:
        raise PermissionError("refusing to decrypt before the ledger records the opening")
    nonce = bytes.fromhex(sealed["nonce_hex"])
    token = sealed["ciphertext"]
    digest = hashlib.sha256(token).hexdigest()
    if digest != sealed["ciphertext_sha256"]:
        raise ValueError("ciphertext hash does not match the published seal")
    return sha256_ctr(key, nonce, token)


class Ledger:
    """Append-only JSONL. The first opening entry is the only one that counts."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def entries(self) -> list[dict]:
        if not self.path.is_file():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def opened(self, payload_id: str) -> bool:
        return any(row.get("event") == "open" and row.get("payload_id") == payload_id for row in self.entries())

    def append(self, row: dict) -> None:
        if row.get("event") == "open" and self.opened(str(row.get("payload_id"))):
            raise PermissionError(f"payload {row.get('payload_id')} was already opened")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
