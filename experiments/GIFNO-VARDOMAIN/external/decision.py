"""Feasibility decision. Fewer than six committed geologies stays Paper 3."""

from __future__ import annotations

from external.sites import MINIMUM_CONFIRMATORY_GEOLOGIES, SITES


def paper_decision(*, timing_measured: bool, custodian: str, cuda: bool, sites=SITES) -> dict:
    committed = [row for row in sites if row["geometry_committed"] and row["tier"] == "candidate_in_domain"]
    geologies = sorted({row["geology_id"] for row in committed})
    blocked = []
    if len(geologies) < MINIMUM_CONFIRMATORY_GEOLOGIES:
        blocked.append(f"committed_in_domain_geologies={len(geologies)}<{MINIMUM_CONFIRMATORY_GEOLOGIES}")
    if not timing_measured:
        blocked.append("solver_timing_unmeasured")
    if not custodian:
        blocked.append("custodian_unassigned")
    if not cuda:
        blocked.append("cuda_unavailable_for_confirmatory_training")
    confirmatory = not blocked
    return {
        "paper": "paper_2_rebuild" if confirmatory else "paper_3_descriptive",
        "rebuild_paper_2": bool(confirmatory),
        "confirmatory_training_opened": False,
        "external_labels_opened": False,
        "committed_geologies": geologies,
        "blocked_reasons": blocked,
        "reported_seed": 0,
        "reported_model": "not_trained",
        "primary_band_hz": [0.1, 10.0],
        "primary_comparator": "corrected_1d_column",
        "secondary_comparator": "P128n128",
        "recorded_motion": "descriptive_appendix_only",
        "length_scaling": "rejected_until_three_rayleigh_fe_pairs_pass",
        "mirror_augmentation": "prespecified",
        "diagnosis": "not_run_learning_curve_inconclusive",
    }
