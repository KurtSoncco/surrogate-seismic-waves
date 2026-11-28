"""Batches that do not let the 7,680 IID runs or the 500 m rows set the step.

Stage 0 and 1: mostly IID, a smaller share of dipping and three-layer.
Stage 2: one quarter IID replay, the rest manifold, equal across widths and
then across strata. ``near_1d_share`` of those manifold slots is drawn from
sections whose lateral log-Vs scatter is below the near-1D cut, so a repeated
column teaches a near-zero residual. The manifold rows keep their own Rayleigh
pair, which is (f1, 3 f1). A pair, when requested, occupies two slots and
shares a draw key.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from torch.utils.data import Sampler

from data import STAGE_UNIFIED

# Half the manifold slots. The repeated-column sections are a small slice of
# the 301, and the flat Case 7 crops are that slice.
NEAR_1D_SHARE = 0.5


class QuotaBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        records: list[dict],
        batch: int,
        stage: int,
        n_batches: int,
        seed: int,
        use_pairs: bool,
        near_1d_share: float = 0.0,
    ):
        self.records = records
        self.batch = int(batch)
        self.stage = int(stage)
        self.n_batches = int(n_batches)
        self.seed = int(seed)
        self.use_pairs = bool(use_pairs)
        self.near_1d_share = float(near_1d_share)
        self.epoch = 0
        self.pools: dict[str, list[int]] = defaultdict(list)
        self.pairs: dict[str, list[tuple[int, int]]] = defaultdict(list)
        seen: set[int] = set()
        for i, rec in enumerate(records):
            key = _pool(rec, stage)
            self.pools[key].append(i)
            if rec.get("pair") and rec["draw"] not in seen:
                mate = next(j for j, other in enumerate(records) if j != i and other.get("pair") and other["draw"] == rec["draw"])
                self.pairs[key].append((i, mate))
                seen.add(rec["draw"])

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch * 10007)
        for _ in range(self.n_batches):
            yield self._batch(rng)

    def _take(self, rng: np.random.Generator, key: str, n: int, used: set[int]) -> list[int]:
        pool = [i for i in self.pools.get(key, []) if i not in used]
        if not pool or n <= 0:
            return []
        pick = rng.choice(pool, size=min(n, len(pool)), replace=False)
        return [int(i) for i in np.atleast_1d(pick)]

    def _take_stratum(
        self, rng: np.random.Generator, width: int, used: set[int], *, near_only: bool = False, exclude_near: bool = False,
    ) -> int | None:
        """One manifold row at ``width``, uniform over strata then over rows."""
        rows = [i for i in self.pools.get(f"man_{width}", []) if i not in used]
        if near_only:
            rows = [i for i in rows if self.records[i].get("near_1d")]
        elif exclude_near:
            rows = [i for i in rows if not self.records[i].get("near_1d")]
        if not rows:
            return None
        by_stratum: dict[str, list[int]] = defaultdict(list)
        for i in rows:
            by_stratum[str(self.records[i].get("stratum", ""))].append(i)
        stratum = str(rng.choice(list(by_stratum)))
        return int(rng.choice(by_stratum[stratum]))

    def _batch(self, rng: np.random.Generator) -> list[int]:
        batch: list[int] = []
        used: set[int] = set()
        if self.use_pairs:
            keys = [k for k, v in self.pairs.items() if v and (self.stage < 2 or k == "iid" or k.startswith("man"))]
            if self.stage >= 2:
                keys = [k for k in keys if k.startswith("man")] or keys
            if keys:
                key = str(rng.choice(keys))
                a, b = self.pairs[key][int(rng.integers(0, len(self.pairs[key])))]
                batch.extend([a, b])
                used.update([a, b])
        if self.stage == STAGE_UNIFIED:
            self._unified(rng, batch, used)
        elif self.stage < 2:
            n_ood = 1 if self.batch >= 4 else 0
            for key in ("ood_dipping", "ood_three_layer"):
                batch.extend(self._take(rng, key, n_ood, used))
                used.update(batch)
            while len(batch) < self.batch:
                got = self._take(rng, "iid", self.batch - len(batch), used)
                if not got:
                    break
                batch.extend(got)
                used.update(got)
        else:
            n_iid = max(1, self.batch // 4) if self.pools.get("iid") else 0
            # the pair already counted if it was IID; fill the replay quota
            n_iid_have = sum(1 for i in batch if self.records[i]["kind"] == "run")
            batch.extend(self._take(rng, "iid", max(0, n_iid - n_iid_have), used))
            used.update(batch)
            widths = [w for w in (500, 1000, 1500, 2000) if self.pools.get(f"man_{w}")]
            turn = 0
            while len(batch) < self.batch and widths:
                w = widths[turn % len(widths)]
                turn += 1
                want_near = self.near_1d_share > 0 and float(rng.random()) < self.near_1d_share
                if want_near:
                    pick = self._take_stratum(rng, w, used, near_only=True)
                else:
                    pick = self._take_stratum(rng, w, used, exclude_near=self.near_1d_share > 0)
                if pick is None:
                    pick = self._take_stratum(rng, w, used)
                if pick is None:
                    widths = [x for x in widths if x != w]
                    continue
                batch.append(pick)
                used.add(pick)
        if len(batch) < self.batch:
            if self.stage == STAGE_UNIFIED:
                allowed = {"iid", "manifold", "cov0"}
                rest = [i for i in range(len(self.records)) if i not in used and self.records[i].get("family") in allowed]
            else:
                rest = [i for i in range(len(self.records)) if i not in used]
            if rest:
                extra = rng.choice(rest, size=min(self.batch - len(batch), len(rest)), replace=False)
                batch.extend(int(i) for i in np.atleast_1d(extra))
        return batch[: self.batch]

    def _unified(self, rng: np.random.Generator, batch: list[int], used: set[int]) -> None:
        """Floor quotas: IID, dipping, three-layer, repeated-column, then manifold widths."""
        # Dipping and three-layer are held out. Their former share falls through to the manifold.
        shares = (("iid", 0.25), ("cov0", 0.10))
        for key, share in shares:
            if not self.pools.get(key):
                continue
            n = int(round(share * self.batch))
            if self.batch >= 8:
                n = max(n, 1)
            got = self._take(rng, key, n, used)
            batch.extend(got)
            used.update(got)
        widths = [w for w in (500, 1000, 1500, 2000) if self.pools.get(f"man_{w}")]
        turn = 0
        while len(batch) < self.batch and widths:
            w = widths[turn % len(widths)]
            turn += 1
            want_near = self.near_1d_share > 0 and float(rng.random()) < self.near_1d_share
            pick = self._take_stratum(rng, w, used, near_only=want_near, exclude_near=not want_near and self.near_1d_share > 0)
            if pick is None:
                pick = self._take_stratum(rng, w, used)
            if pick is None:
                widths = [x for x in widths if x != w]
                continue
            batch.append(pick)
            used.add(pick)


def _pool(rec: dict, stage: int) -> str:
    if rec["kind"] == "section":
        return f"man_{int(rec.get('lx', rec.get('width', 0)))}"
    return str(rec.get("family", "iid"))


def batches_per_epoch(records: list[dict], batch: int, stage: int, fixed: int | None = None) -> int:
    if fixed is not None:
        return max(1, int(fixed))
    if stage == STAGE_UNIFIED:
        counts = defaultdict(int)
        for rec in records:
            counts[_pool(rec, stage)] += 1
        if not counts:
            return 1
        per = max(1, batch // 4)
        return max(1, int(np.ceil(max(counts.values()) / per)))
    if stage < 2:
        n_iid = sum(1 for r in records if r.get("family") == "iid")
        per = max(1, batch - 2)
        return max(1, int(np.ceil(n_iid / per)))
    n_man = sum(1 for r in records if r["kind"] == "section")
    per = max(1, batch - max(1, batch // 4))
    return max(1, int(np.ceil(n_man / per)))
