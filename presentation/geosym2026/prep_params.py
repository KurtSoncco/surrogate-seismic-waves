"""Per-run site parameters of the horizontal and dipping campaigns, one CSV.

    python prep_params.py
"""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd

from figstyle import DATA, REPO

H5 = {
    "iid": REPO / "data" / "gifno_screen" / "h5",
    "ood_dipping": REPO / "data" / "gifno_screen" / "ood_dipping" / "h5",
}
CACHE = REPO / "data" / "vardomain" / "cache"
KEYS = ("Vs1", "Vs2", "CoV", "rH", "aHV", "f0_effective", "H")


def main() -> None:
    rows = []
    for family, root in H5.items():
        split = np.load(CACHE / family / "split.npy")
        group = np.load(CACHE / family / "group.npy")
        run_index = np.load(CACHE / family / "run_index.npy")
        for i, run in enumerate(run_index):
            with h5py.File(root / f"run_{int(run)}.h5", "r") as handle:
                attrs = handle["params"].attrs
                row = {k: float(attrs[k]) for k in KEYS if k in attrs}
            rows.append({"family": family, "i": i, "run": int(run), "split": str(split[i]),
                         "group": int(group[i]), **row})
    frame = pd.DataFrame(rows)
    DATA.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DATA / "run_params.csv", index=False)
    print(frame.groupby("family")[list(KEYS)].median().to_string())
    print("median f0 all runs", frame["f0_effective"].median())
    print("cache index equals run number:", bool((frame["i"] == frame["run"]).all()))


if __name__ == "__main__":
    main()
