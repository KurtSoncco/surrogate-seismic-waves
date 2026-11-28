"""Frozen inference geometry. Truncated windows never enter the confirmatory mean."""

from __future__ import annotations

CONFIRMATORY_WIDTH_M = 2000
STATION_STRIDE_M = 20
EDGE_MARGIN_M = 5


def confirmatory_stations(length_m: float, *, width_m: int = CONFIRMATORY_WIDTH_M,
                          stride_m: int = STATION_STRIDE_M) -> dict:
    """Stations whose centered window lies entirely inside the section."""
    length = float(length_m)
    half = float(width_m) / 2.0
    if length < float(width_m):
        return {
            "confirmatory_x_m": [],
            "truncated_x_m": [float(x) for x in range(EDGE_MARGIN_M, int(length) - EDGE_MARGIN_M, stride_m)],
            "width_m": int(width_m),
            "stride_m": int(stride_m),
            "stitch": "none; overlapping windows are not averaged",
        }
    start = int(half)
    stop = int(length - half)
    kept = [float(x) for x in range(start, stop + 1, stride_m)]
    truncated = [float(x) for x in range(EDGE_MARGIN_M, start, stride_m)]
    truncated.extend(float(x) for x in range(stop + stride_m, int(length) - EDGE_MARGIN_M, stride_m))
    return {
        "confirmatory_x_m": kept,
        "truncated_x_m": truncated,
        "width_m": int(width_m),
        "stride_m": int(stride_m),
        "stitch": "none; one centered window per confirmatory station",
    }
