"""Metadata eligibility. A section is never admitted because its score looks good."""

from __future__ import annotations

IN_DOMAIN = {
    "width_m": (500.0, 2000.0),
    "depth_m_max": 250.0,
    "encoder_cells_max": 256,
    "flat_surface": True,
}


def classify_section(meta: dict, envelope: dict) -> dict:
    """Return in_domain, extrapolation, or ineligible from metadata alone."""
    reasons = []
    width = float(meta["width_m"])
    depth = float(meta["depth_m"])
    cells = int(meta.get("encoder_cells", round(depth)))
    if width < IN_DOMAIN["width_m"][0] or width > IN_DOMAIN["width_m"][1]:
        reasons.append("width outside 500-2000 m")
    if depth > IN_DOMAIN["depth_m_max"] or cells > IN_DOMAIN["encoder_cells_max"]:
        reasons.append("depth outside the encoder")
    if not bool(meta.get("flat_surface", False)):
        reasons.append("topography is not a flat surface")
    if not bool(meta.get("elastic_half_space", False)):
        reasons.append("no defensible elastic half-space")
    vs_min = float(meta["vs_min"])
    vs_max = float(meta["vs_max"])
    if vs_min < float(envelope["vs_min"]) or vs_max > float(envelope["vs_max"]):
        reasons.append("Vs outside the frozen training envelope")
    impedance = float(meta.get("impedance_ratio", vs_max / max(vs_min, 1.0)))
    if impedance > float(envelope["impedance_ratio_max"]):
        reasons.append("impedance ratio outside the frozen training envelope")
    tier = "in_domain" if not reasons else "extrapolation"
    if meta.get("blocked"):
        tier = "ineligible"
        reasons.append(str(meta.get("blocked_reason") or "blocked by the sampling frame"))
    return {"tier": tier, "reasons": reasons, "confirmatory": tier == "in_domain"}
