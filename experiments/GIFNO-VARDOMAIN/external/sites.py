"""Metadata catalog. Geometry is committed only when a local hashed file exists."""

from __future__ import annotations

SITES = (
    {
        "geology_id": "mygdonian",
        "name": "Mygdonian/Volvi",
        "tier": "candidate_in_domain",
        "geometry_committed": False,
        "reason": "waveforms are public; the continuous 2D interfaces are not a local hashed grid",
        "citation": "10.1016/S0267-7261(99)00025-1",
    },
    {
        "geology_id": "nice",
        "name": "Nice sedimentary basin",
        "tier": "candidate_in_domain",
        "geometry_committed": False,
        "reason": "published section only; no downloadable mesh",
        "citation": "10.1093/gji/ggu256",
    },
    {
        "geology_id": "thorndon",
        "name": "Thorndon basin",
        "tier": "candidate_in_domain",
        "geometry_committed": False,
        "reason": "reconstructable from public sources, but no local mesh has been hashed",
        "citation": "10.5459/bnzsee.54.1.21-30",
    },
    {
        "geology_id": "garner_valley",
        "name": "Garner Valley",
        "tier": "candidate_in_domain",
        "geometry_committed": False,
        "reason": "observations are public; the pseudo-3D grid was not located",
        "citation": "nees.ucsb.edu/facilities/GVDA",
    },
    {
        "geology_id": "parkway",
        "name": "Parkway, Wainuiomata",
        "tier": "candidate_in_domain",
        "geometry_committed": False,
        "reason": "no public continuous grid; observed response is demonstrably 3D",
        "citation": "10.1785/bssa0890030718",
    },
    {
        "geology_id": "bagneres",
        "name": "Bagneres-de-Bigorre",
        "tier": "candidate_in_domain",
        "geometry_committed": False,
        "reason": "profiles are in the paper, not a packaged mesh",
        "citation": "10.1785/0120100293",
    },
    {
        "geology_id": "grenoble",
        "name": "Grenoble basin",
        "tier": "extrapolation",
        "geometry_committed": False,
        "reason": "deeper than 250 m and strongly 3D",
        "citation": "10.1016/j.soildyn.2012.02.001",
    },
    {
        "geology_id": "usgs_scec_slices",
        "name": "USGS Bay Area or SCEC slices",
        "tier": "extrapolation",
        "geometry_committed": False,
        "reason": "slices of one regional model are one geology, and deep basins are outside the domain",
        "citation": "regional velocity models",
    },
)

FALLBACK_ORDER = ("mygdonian", "nice", "thorndon", "garner_valley", "parkway", "bagneres")
MINIMUM_CONFIRMATORY_GEOLOGIES = 6
TARGET_CONFIRMATORY_GEOLOGIES = 8
FAMILY_MINIMUM = 3
