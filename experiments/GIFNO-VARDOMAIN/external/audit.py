"""Confirmatory comparisons. Sites are the independent unit, and the order is fixed."""

from __future__ import annotations

import numpy as np

from external.sites import FAMILY_MINIMUM, MINIMUM_CONFIRMATORY_GEOLOGIES


def equal_geology_mean(site_errors: dict[str, float]) -> float:
    if not site_errors:
        raise ValueError("no geologies")
    return float(np.mean(list(site_errors.values())))


def geology_interval(site_errors: dict[str, float], seed: int = 0, n_boot: int = 2000) -> dict:
    """Resample geologies. Station resampling is not a substitute."""
    values = np.asarray(list(site_errors.values()), dtype=np.float64)
    if values.size < MINIMUM_CONFIRMATORY_GEOLOGIES:
        return {
            "mean": float(values.mean()) if values.size else float("nan"),
            "interval": [float("nan"), float("nan")],
            "confirmatory": False,
            "reason": f"{values.size} geologies is below {MINIMUM_CONFIRMATORY_GEOLOGIES}",
        }
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, values.size, size=(n_boot, values.size))].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"mean": float(values.mean()), "interval": [float(low), float(high)], "confirmatory": True, "reason": ""}


def family_gate(family_sites: dict[str, dict[str, float]]) -> dict:
    """A family is inferential only with at least three independent geologies."""
    ready = {name: errors for name, errors in family_sites.items() if len(errors) >= FAMILY_MINIMUM}
    descriptive = sorted(set(family_sites) - set(ready))
    return {"inferential": sorted(ready), "descriptive_only": descriptive}


def confirmatory_decision(
    *,
    model_error: dict[str, float],
    column_error: dict[str, float],
    reference_error: dict[str, float] | None,
    adequacy: float,
    margin: float,
    noninferiority: float,
) -> dict:
    """Test the column first. The old model is tested only after that gate passes."""
    if not model_error:
        raise ValueError("no site errors")
    model_mean = equal_geology_mean(model_error)
    column_mean = equal_geology_mean(column_error)
    primary_delta = model_mean - column_mean
    primary_pass = model_mean <= adequacy and primary_delta <= -abs(margin)
    worse_family = []
    if reference_error and primary_pass:
        for site, value in model_error.items():
            if value - reference_error[site] > noninferiority:
                worse_family.append(site)
    secondary_pass = primary_pass and reference_error is not None and not worse_family and (
        model_mean < equal_geology_mean(reference_error)
    )
    return {
        "primary_comparator": "corrected_1d_column",
        "primary_pass": bool(primary_pass),
        "primary_delta": primary_delta,
        "secondary_tested": bool(primary_pass and reference_error is not None),
        "secondary_pass": bool(secondary_pass),
        "sites_worse_than_reference": worse_family,
        "model_mean": model_mean,
        "column_mean": column_mean,
    }
