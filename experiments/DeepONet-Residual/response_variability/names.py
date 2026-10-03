"""Readable method names and Nature-safe colors (Wong 2011)."""

from __future__ import annotations

OPENSEES = "OpenSees 2-D"
GINO = "GINO"
HASKELL_NOMINAL = "1D Base Case"
HASKELL_COLUMN = "1D column"
TORO = "Toro Vs"
TORO_FIXED = "Toro fixed H"
TORO_DIP = "Toro dip"
PASSERI = "Passeri tts"
PASSERI_FIXED = "Passeri fixed H"
PASSERI_DIP = "Passeri dip"
PRETELL = "Pretell median"
PRETELL_P84 = "Pretell percentile"
DMULT = "Dmult"
DMULT_P84 = "Dmult p84"

# SOTA ranking uses Pretell median + percentile only. ``HASKELL_COLUMN`` is
# per-recorder 1D Haskell from the signed cache, not the 200-column Pretell arm.
COMPARE_METHODS = (GINO, HASKELL_NOMINAL)
SEISKIT_METHODS = (TORO, PASSERI, PRETELL, PRETELL_P84, DMULT, DMULT_P84)
ALL_METHODS = (OPENSEES, *COMPARE_METHODS, *SEISKIT_METHODS)

# Wong / Nature colorblind palette. OpenSees is the black reference.
METHOD_COLORS = {
    OPENSEES: "#000000",
    GINO: "#0072B2",
    HASKELL_NOMINAL: "#999999",
    HASKELL_COLUMN: "#009E73",
    TORO: "#D55E00",
    TORO_FIXED: "#D55E00",
    TORO_DIP: "#EE6677",
    PASSERI: "#CC79A7",
    PASSERI_FIXED: "#CC79A7",
    PASSERI_DIP: "#882255",
    PRETELL: "#E69F00",
    PRETELL_P84: "#6B3A0F",
    DMULT: "#56B4E9",
    DMULT_P84: "#2B6A8A",
}

METHOD_LINESTYLES = {
    OPENSEES: "-",
    GINO: "-",
    HASKELL_NOMINAL: "-.",
    HASKELL_COLUMN: "--",
    TORO: (0, (3, 1, 1, 1)),
    TORO_FIXED: (0, (7, 2.4)),
    TORO_DIP: (0, (2.2, 1.4)),
    PASSERI: ":",
    PASSERI_FIXED: (0, (9, 2.2, 2.2, 2.2)),
    PASSERI_DIP: (0, (1.2, 1.3)),
    PRETELL: "-.",
    PRETELL_P84: (0, (6, 1.4)),
    DMULT: (0, (1, 1)),
    DMULT_P84: (0, (4, 1.5, 1, 1.5)),
}

METHOD_ZORDER = {
    OPENSEES: 4,
    GINO: 5,
    HASKELL_NOMINAL: 2,
    HASKELL_COLUMN: 3,
    TORO: 2,
    PASSERI: 2,
    PRETELL: 3,
    PRETELL_P84: 6,
    DMULT: 2,
    DMULT_P84: 3,
}

TF_KEYS = {
    OPENSEES: "tf_opensees",
    GINO: "tf_gino",
    HASKELL_NOMINAL: "tf_haskell_nominal",
    HASKELL_COLUMN: "tf_haskell_column",
    TORO: "tf_toro",
    TORO_FIXED: "tf_toro_fixed",
    TORO_DIP: "tf_toro_dip",
    PASSERI: "tf_passeri",
    PASSERI_FIXED: "tf_passeri_fixed",
    PASSERI_DIP: "tf_passeri_dip",
    PRETELL: "tf_pretell",
    PRETELL_P84: "tf_pretell_p84",
    DMULT: "tf_dmult",
    DMULT_P84: "tf_dmult_p84",
}
