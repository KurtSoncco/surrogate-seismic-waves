"""Readable method names and Nature-safe colors (Wong 2011)."""

from __future__ import annotations

OPENSEES = "OpenSees 2-D"
GINO = "GINO"
HASKELL_NOMINAL = "1D Base Case"
HASKELL_COLUMN = "Pretell's approach"
TORO = "Toro Vs"
PASSERI = "Passeri tts"
PRETELL = "Pretell"
PRETELL_P84 = "Pretell p84"

COMPARE_METHODS = (GINO, HASKELL_NOMINAL, HASKELL_COLUMN)
SEISKIT_METHODS = (TORO, PASSERI, PRETELL, PRETELL_P84)
ALL_METHODS = (OPENSEES, *COMPARE_METHODS, *SEISKIT_METHODS)

# Wong / Nature colorblind palette. OpenSees is the black reference.
METHOD_COLORS = {
    OPENSEES: "#000000",
    GINO: "#0072B2",
    HASKELL_NOMINAL: "#999999",
    HASKELL_COLUMN: "#009E73",
    TORO: "#D55E00",
    PASSERI: "#CC79A7",
    PRETELL: "#E69F00",
    PRETELL_P84: "#6B3A0F",
}

METHOD_LINESTYLES = {
    OPENSEES: "-",
    GINO: "-",
    HASKELL_NOMINAL: "-.",
    HASKELL_COLUMN: "--",
    TORO: (0, (3, 1, 1, 1)),
    PASSERI: ":",
    PRETELL: "-.",
    PRETELL_P84: (0, (6, 1.4)),
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
}

TF_KEYS = {
    OPENSEES: "tf_opensees",
    GINO: "tf_gino",
    HASKELL_NOMINAL: "tf_haskell_nominal",
    HASKELL_COLUMN: "tf_haskell_column",
    TORO: "tf_toro",
    PASSERI: "tf_passeri",
    PRETELL: "tf_pretell",
    PRETELL_P84: "tf_pretell_p84",
}
