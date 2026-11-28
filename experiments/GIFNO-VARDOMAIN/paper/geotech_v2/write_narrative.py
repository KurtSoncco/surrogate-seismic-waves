#!/usr/bin/env python3
"""Write the geotechnical narrative from the frozen ladder and the Phase 1 audit.

Every quoted number is read from analysis/frozen, analysis/audit, or, where a
definition difference has to be named, analysis/G0.md. The script does not
rescore a model and does not mix a mean of station-wise skills with the skill
of a pooled error.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parents[1] / "analysis"
FROZEN = ANALYSIS / "frozen"
AUDIT = ANALYSIS / "audit"


def num(value, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def pct(value) -> str:
    return f"{100.0 * float(value):.2f}%"


def one(frame: pd.DataFrame, **query) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for key, value in query.items():
        mask &= frame[key] == value
    hit = frame[mask]
    if len(hit) != 1:
        raise SystemExit(f"expected one row for {query}, found {len(hit)}")
    return hit.iloc[0]


def skill_text(row: pd.Series) -> str:
    if row.skill_status != "pooled" or pd.isna(row.pooled_skill):
        return "suppressed by the 0.0136 proposal"
    return num(row.pooled_skill, 4)


def main() -> None:
    ladder = pd.read_csv(FROZEN / "ladder.csv")
    treasure = pd.read_csv(FROZEN / "ladder_treasure.csv")
    baselines = pd.read_csv(FROZEN / "baselines.csv")
    budget = pd.read_csv(FROZEN / "error_budget.csv")
    consistency = pd.read_csv(FROZEN / "consistency_tests.csv")
    epoch = pd.read_csv(FROZEN / "epoch_neighborhood.csv")
    splits = pd.read_csv(AUDIT / "splits.csv")
    census = pd.read_csv(AUDIT / "census.csv")
    noise = json.loads((AUDIT / "noise_floor.json").read_text())
    ladder_prov = json.loads((FROZEN / "ladder.provenance.json").read_text())
    treasure_prov = json.loads((FROZEN / "ladder_treasure.provenance.json").read_text())
    budget_prov = json.loads((FROZEN / "error_budget.provenance.json").read_text())
    consistency_prov = json.loads((FROZEN / "consistency_tests.provenance.json").read_text())
    epoch_prov = json.loads((FROZEN / "epoch_neighborhood.provenance.json").read_text())
    representation = pd.read_csv(AUDIT / "representation.csv")
    case7 = pd.read_csv(FROZEN / "case7_column.csv")
    cache = pd.read_csv(FROZEN / "cache_engine_check.csv")
    identity = pd.read_csv(FROZEN / "treasure_skill_identity.csv")
    rationale = pd.read_csv(AUDIT / "domain_rationale.csv")
    dtype_stamp = pd.read_csv(FROZEN / "dtype_stamp.csv")
    exemplars = pd.read_csv(FROZEN / "heatmap_exemplars.csv")

    floor = float(one(consistency, test="homogeneous_column", band="low").reference)
    iid_mid = one(ladder, rung="iid_test", band="mid")
    dip_mid = one(ladder, rung="dipping_test", band="mid")
    _require_positive(ladder, baselines, ("iid_test", "dipping_test"), ("mid", "high"))
    tl_mid = one(ladder, rung="three_layer_test", band="mid")
    tl_prior = one(baselines, rung="three_layer_test", band="mid", baseline="g0")
    tl_real = one(baselines, rung="three_layer_test", band="mid", baseline="realization_floor")
    if not (float(tl_prior.g_log_l1) < floor and float(tl_real.g_log_l1) > float(tl_prior.g_log_l1)):
        raise SystemExit("three-layer mid-band no longer matches the G2 exclusion")
    ti_low = one(treasure, rung="treasure_island", band="low")
    if ti_low.accuracy_claim != "not_an_accuracy_result":
        raise SystemExit("Treasure Island low band is marked as an accuracy result")

    lines = _narrative(
        ladder, treasure, baselines, budget, consistency, epoch, splits, census, noise,
        representation, ladder_prov, treasure_prov, budget_prov, consistency_prov, epoch_prov,
        floor, iid_mid, dip_mid, tl_mid, tl_prior, tl_real,
        case7, cache, identity, rationale, dtype_stamp, exemplars,
    )
    text = "\n".join(lines) + "\n"
    (HERE / "README.md").write_text(text)
    _sections(
        ladder, treasure, baselines, consistency, epoch, splits, noise, floor,
        iid_mid, treasure_prov, epoch_prov, exemplars, identity,
    )
    (HERE.parent / "README.md").write_text(
        "# Superseded\n\n"
        "This page is superseded by [`geotech_v2/README.md`](geotech_v2/README.md).\n\n"
        "The previous stage-2 narrative is not a current headline. It mixed mean stationwise "
        "skills with pooled-error skills, mixed a practice box across priors, and carried "
        "Case 7 and SVLAB comparisons from the old half-space engine. Those comparisons are "
        "quarantined. The generated narrative is `geotech_v2/write_narrative.py`.\n"
    )
    print(f"wrote {HERE / 'README.md'}")


def _require_positive(ladder, baselines, rungs, bands) -> None:
    for rung in rungs:
        for band in bands:
            model = one(ladder, rung=rung, band=band)
            center = float(one(baselines, rung=rung, band=band, baseline="center_column").g_log_l1)
            pretell = float(one(baselines, rung=rung, band=band, baseline="pretell_geomean").g_log_l1)
            prior = float(one(baselines, rung=rung, band=band, baseline="g0").g_log_l1)
            point = float(model.g_log_l1)
            upper = float(model.ci_high)
            if not (point < center and upper < center and point < pretell and upper < pretell and point < prior):
                raise SystemExit(f"pooled comparison failed for {rung} {band}")


def _cited(frame: pd.DataFrame, name: str) -> int:
    return int(one(frame, name=name).value)


def _narrative(
    ladder, treasure, baselines, budget, consistency, epoch, splits, census, noise,
    representation, ladder_prov, treasure_prov, budget_prov, consistency_prov, epoch_prov,
    floor, iid_mid, dip_mid, tl_mid, tl_prior, tl_real,
    case7, cache, identity, rationale, dtype_stamp, exemplars,
) -> list[str]:
    iid_n = int(iid_mid.n_records)
    iid_g = int(iid_mid.n_masters)
    dip_n = int(dip_mid.n_records)
    dip_g = int(dip_mid.n_masters)
    epoch_row = one(splits, table="epoch_rule")
    distance = one(splits, table="treasure_distance")
    ti_pod = one(representation, table="pod_floor", pool="treasure_island", band="low")
    leakage = noise["record_length_leakage"]
    dtype_row = one(dtype_stamp, name="max_abs_g_log_l1")
    iid_cache = one(cache, family="iid", station_role="all_checked")
    dip_cache = one(cache, family="ood_dipping", station_role="all_checked")
    tl_cache = one(cache, family="ood_three_layer", station_role="all_checked")
    lines = [
        "# Compatibility on a 500 m strip",
        "",
        (
            "Three claims are supported. The learned correction is empirically compatible with the column "
            "on windows that repeat a 500 m strip. On the IID and dipping tests, that correction reduces "
            "pooled absolute log error in 0.4–10 Hz. Broad lateral accuracy, and accuracy on a real section "
            "below 0.4 Hz, are not established. A padded campaign at 500, 1000, 1500, and 2000 m is future work. "
            "This revision does not run it."
        ),
        "",
        "This page is generated from the frozen ladder, baselines, error budget, consistency tests, epoch neighborhood, Case 7 column rescore, and the Phase 1 audit. Provenance sidecars record the checkpoint, the float32 forward, and the aggregation. A definition that differs from an earlier draft is named from `analysis/G0.md` and is not reprinted as a second headline.",
        "",
        "The surface transfer function is written",
        "",
        r"$$|\mathrm{TF}| = |\mathrm{TF}_{\mathrm{column}}|\,e^{g}.$$",
        "",
        "The column is the elastic half-space outcrop of the soil profile under the station, with the damping OpenSees applied, matched to Rayleigh damping at the two frequencies of that record. The network returns only g. Setting g to zero is an identity of this residual parameterization: the prediction is then the column. It is not a guarantee that the network emits zero.",
        "",
        f"The reported model is P128n128, epoch {int(ladder_prov['epoch'])}, {ladder_prov['dtype']}. The ladder pools every labeled station and resamples Sobol groups (`bootstrap_unit={ladder_prov['bootstrap_unit']}`). Skill, where it is printed, is the skill of that pooled mean. Bands are 0.1–0.4 Hz, 0.4–2.5 Hz, 2.5–10 Hz, and 0.1–10 Hz.",
        "",
        (
            f"The reported checkpoint forward is float32. Training config amp is on. "
            f"evaluate.load_model binds bfloat16 autocast only on CUDA. "
            f"The float32 comparison is one centered {int(dtype_row.width_m)} m window on {dtype_row.device}, "
            f"autocast stayed off, and the maximum absolute g log-L1 difference is {num(dtype_row.value, 6)}. "
            "A GPU bfloat16 sweep of that autocast was not run."
        ),
        "",
        (
            f"On the frozen float16 cache, the maximum absolute log gap versus surface_over_outcrop is "
            f"{num(iid_cache.max_abs_log, 6)} on IID, {num(dip_cache.max_abs_log, 6)} on dipping, and "
            f"{num(tl_cache.max_abs_log, 6)} on three-layer. cache/meta.json does not name the engine. "
            "The check is one center column and one edge column from each test group, not every station."
        ),
        "",
        "## Headline",
        "",
        (
            f"Paper 2 is positive for the IID test ({iid_n} records, {iid_g} Sobol groups) and the dipping test "
            f"({dip_n} records, {dip_g} groups) in 0.4–10 Hz: pooled g log-L1 and the upper end of the master "
            f"bootstrap lie below the center-column outcrop and below the 9-column Pretell geometric mean. "
            f"IID mid-band pooled g log-L1 is {num(iid_mid.g_log_l1)}. This is not a lateral-accuracy result. "
            f"The 0.1–0.4 Hz band is absolute error only. Three-layer is outside the claim (mid-band prior "
            f"{num(tl_prior.g_log_l1)} under the {num(floor, 4)} skill floor, harm on {pct(tl_mid.harm_rate)} of records, "
            f"realization floor {num(tl_real.g_log_l1)} above the 1D error). Treasure Island below 0.4 Hz is not "
            "an accuracy result. Manifold widths and synthetic 1000/1500 m windows are compatibility. The training strip is 500 m."
        ),
        "",
        "The same pooled point estimates on those two tests and those two bands also lie below g = 0, the stored column prior. Intervals below are the master-bootstrap interval of the pooled mean.",
        "",
        "| Test | Band | P128n128 (interval) | g = 0 | Center column | Pretell, 9 columns | Harm | Pooled skill |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rung, label in (("iid_test", "IID"), ("dipping_test", "Dipping")):
        for band, band_name in (("mid", "0.4–2.5 Hz"), ("high", "2.5–10 Hz")):
            model = one(ladder, rung=rung, band=band)
            lines.append(
                f"| {label} | {band_name} | {num(model.g_log_l1)} ({num(model.ci_low)}–{num(model.ci_high)}) | "
                f"{num(one(baselines, rung=rung, band=band, baseline='g0').g_log_l1)} | "
                f"{num(one(baselines, rung=rung, band=band, baseline='center_column').g_log_l1)} | "
                f"{num(one(baselines, rung=rung, band=band, baseline='pretell_geomean').g_log_l1)} | "
                f"{pct(model.harm_rate)} | {skill_text(model)} |"
            )
    lines += [
        "",
        f"IID mid-band harm is {pct(iid_mid.harm_rate)} of {iid_n} records. Dipping mid-band harm is {pct(dip_mid.harm_rate)} of {dip_n} records. Harm is the share of records whose pooled g log-L1 is worse than g = 0.",
        "",
        f"Ladder provenance: checkpoint epoch {int(ladder_prov['epoch'])}, {ladder_prov['dtype']}, aggregation `{ladder_prov['aggregation_unit']}`, skill reduction `{ladder_prov['skill_reduction']}`. The Pretell mean uses the nine columns listed in `ladder.provenance.json`. Sample-id hash `{ladder_prov['sample_id_hash'][:12]}`.",
        "",
        "## What is outside that sentence",
        "",
        "Three-layer stays out. Its mid-band prior sits under the proposal floor, so the ladder suppresses skill. About one third of those records are harmed, and the realization floor exceeds the 1D error. Absolute mid-band g log-L1 is "
        f"{num(tl_mid.g_log_l1)} (interval {num(tl_mid.ci_low)}–{num(tl_mid.ci_high)}, {int(tl_mid.n_records)} records, {int(tl_mid.n_masters)} groups).",
        "",
        "The 0.1–0.4 Hz band is reported as absolute error. It is not part of the positive test.",
        "",
        "| Test | Pooled g log-L1 | Interval | g = 0 |",
        "| --- | --- | --- | --- |",
    ]
    for rung, label in (("iid_test", "IID"), ("dipping_test", "Dipping"), ("three_layer_test", "Three-layer")):
        model = one(ladder, rung=rung, band="low")
        prior = one(baselines, rung=rung, band="low", baseline="g0")
        lines.append(
            f"| {label} | {num(model.g_log_l1)} | {num(model.ci_low)}–{num(model.ci_high)} | {num(prior.g_log_l1)} |"
        )
    lines += [
        "",
        f"The measured 1D-convention floor is {noise['measured_noise_floor']:.6f} on 2.5–10 Hz, a lower bound on max(FE floor, 1D-convention uncertainty) because the FE floor was not measured (`noise_floor.json`). Halving the 15 s Treasure Island record moves log|TF| by {leakage['low']['full_qvs.h5@7.490000000223517']:.3f} (0.1–0.4 Hz), {leakage['mid']['full_qvs.h5@7.490000000223517']:.3f} (0.4–2.5 Hz), and {leakage['high']['full_qvs.h5@7.490000000223517']:.3f} (2.5–10 Hz).",
        "",
        "## Practice baselines",
        "",
        "The in-family comparison is the table above: pooled g log-L1 against g = 0, the center-column elastic outcrop, and a 9-column geometric-mean outcrop. Both outcrops are elastic half-space calculations with Q(Vs) matched to Rayleigh damping at the record frequencies. Cached rigid-base arms are not the comparison.",
        "",
        "Practice Pearson is of log|TF| only. G0 names the difference: the previous box used Pearson of linear |TF| on the full spectrum, then a median across records. That box is not reprinted, and its Pearson values are not placed beside this pooled log-L1.",
        "",
        "The bars in `03_state_of_practice/central.pdf` are that elastic pooled comparison, read from `ladder.csv` and `baselines.csv`.",
        "",
        "## Compatibility, and the coverage of the training strip",
        "",
        "The training strip is 500 m. A window of 1000 m or 1500 m on the IID, dipping, and three-layer tests repeats the edge column of that strip. The change in pooled g log-L1 is context invariance. It is compatibility with the column, not a lateral-accuracy result. `04_width_and_manifold/central.pdf` plots that invariance from `consistency_tests.csv`. The previous width-skill curve is not in the main text.",
        "",
        "Mid-band pooled g log-L1 when the same stations are placed in a wider window:",
        "",
        "| Family | 500 m | 1000 m | 1500 m |",
        "| --- | --- | --- | --- |",
    ]
    for family, label in (("iid", "Horizontal"), ("ood_dipping", "Dipping"), ("ood_three_layer", "Three-layer")):
        cells = []
        for width in (500, 1000, 1500):
            row = one(consistency, test="context_invariance", family=family, width_m=width, band="mid")
            cells.append(num(row.value))
        lines.append(f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines += [
        "",
        "## Case 7, corrected column",
        "",
        "The controlled width diagnostic is OpenSees versus the corrected elastic column on the existing Case 7 H5s. "
        "The network was not scored. There is no CUDA in this environment, and a new forward was not run. "
        "full_svlab.h5 was not read. Pre-fix Pearson tables are not reprinted.",
        "",
        (
            "Hallal, Cox, and Vantassel (JGGE 2023) and Hallal's dissertation are the domain rationale already used by this project: "
            f"variability along at least {_cited(rationale, 'hallal_fundamental_mode_m')} m for the fundamental-mode drop, about "
            f"{_cited(rationale, 'hallal_other_basin_effects_m')} m for the other basin effects, and ground out to about "
            f"{_cited(rationale, 'hallal_ground_from_sensor_m')} m from the sensor. Those lengths were not computed in this repository."
        ),
        "",
        (
            f"Mean |log| gap on stations every {int(one(case7, name='flat_west', width_m=500, band='mid').stride_m)} m. "
            f"FFT window {num(one(case7, name='flat_west', width_m=500, band='mid').fft_window_s, 2)} s. "
            f"Input scale {num(one(case7, name='flat_west', width_m=500, band='mid').scale, 4)}, the acceleration value recorded with that window."
        ),
        "",
        "| Crop | 500 m, 0.1–0.4 Hz | 1500 m, 0.1–0.4 Hz | 500 m, 0.4–2.5 Hz | 1500 m, 0.4–2.5 Hz |",
        "| --- | --- | --- | --- | --- |",
    ]
    for crop_name, crop_label in (
        ("flat_west", "Flat west"),
        ("center", "Center"),
        ("thinning", "Thinning"),
        ("ramp", "Ramp"),
        ("flat_east", "Flat east"),
    ):
        cells = []
        for width, band in ((500, "low"), (1500, "low"), (500, "mid"), (1500, "mid")):
            cells.append(num(one(case7, name=crop_name, width_m=width, band=band).mean_abs_log, 4))
        lines.append(f"| {crop_label} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
    lines += [
        "",
        "The low-band gap is not monotone in width. A 2000 m row is a window of full_qvs.h5. "
        "Nonzero mesh_shift_m means the window was slid onto the 4000 m mesh and meets the domain edge. "
        "Open markers on `04_width_and_manifold/case7_column.pdf` are those rows. They are not a padded 2000 m domain, and they are not a network score.",
        "",
    ]
    iid_census = one(census, campaign="iid")
    dip_census = one(census, campaign="ood_dipping")
    tl_census = one(census, campaign="ood_three_layer")
    man_census = one(census, campaign="manifold cache sections")
    lines += [
        "",
        f"Census: IID {int(iid_census.n_records)} records in {int(iid_census.n_groups)} groups, split {int(iid_census.n_train)}/{int(iid_census.n_val)}/{int(iid_census.n_test)}, a {int(iid_census.width_m)} m strip of a {int(iid_census.mesh_m)} m mesh, depth up to {num(iid_census.depth_m_max, 0)} m. Dipping {int(dip_census.n_records)} records, split {int(dip_census.n_train)}/{int(dip_census.n_val)}/{int(dip_census.n_test)}. Three-layer {int(tl_census.n_records)} records, the same split. Manifold cache {int(man_census.n_records)} sections, split {int(man_census.n_train)}/{int(man_census.n_val)}/{int(man_census.n_test)}/{int(man_census.n_blind)}.",
        "",
        "Coverage is the per-record table `coverage.csv`, with the summary in `coverage.provenance.json`. Every IID and OOD training window is 500 m, so a low-frequency lateral question sits where that density has no other width.",
        "",
        "The coverage figure is `04_width_and_manifold/coverage.png`, copied from the audit.",
        "",
        "Manifold sections are pad-cropped. Troughs are absent from the cache that was scored. Blind counts per width are 2–4 sections. Those rows are compatibility.",
        "",
        "| Split | Width | n records | Mid-band pooled g log-L1 | Interval |",
        "| --- | --- | --- | --- | --- |",
    ]
    for rung in (
        "manifold_val", "manifold_blind",
        "manifold_blind_w500", "manifold_blind_w1000", "manifold_blind_w1500", "manifold_blind_w2000",
    ):
        row = one(ladder, rung=rung, band="mid")
        width = "pooled" if row.width_m == "pooled" else f"{int(float(row.width_m))} m"
        lines.append(
            f"| {row.split} | {width} | {int(row.n_records)} | {num(row.g_log_l1)} | {num(row.ci_low)}–{num(row.ci_high)} |"
        )
    val = one(ladder, rung="manifold_val", band="mid")
    blind = one(ladder, rung="manifold_blind", band="mid")
    auc = one(splits, table="adversarial", comparison="manifold train vs manifold blind")
    lines += [
        "",
        f"On this pooled reduction the blind mid-band g log-L1 is {num(blind.g_log_l1)} and the validation mid-band g log-L1 is {num(val.g_log_l1)}. Blind was not in the selection score. A logistic classifier separates manifold blind from manifold train at AUC {num(auc.auc_mean, 3)} on {int(auc.n_b)} sections (`splits.csv`). The blind pool is small, and the per-width rows are not a resolved width result. `06_reliability_and_broadband/central.pdf` shows the same pooled means.",
        "",
        "## Error budget",
        "",
        "For each family and band the budget is the 1D-prior error, the error left after P128n128, the realization floor, and the POD floor. The headline ratio is (1D error − model error) / (1D error − max of the measured floors). The FE floor is unmeasured, so every finite ratio is provisional. A nonpositive gap is omitted.",
        "",
        "On IID and dipping, in the mid band, the high band, and overall, the realization floor exceeds the 1D error, so those ratios are omitted. Finite ratios that remain are marked `provisional_fe_unmeasured`. The figure `06_reliability_and_broadband/error_budget.pdf` is drawn from `error_budget.csv` only. The FE floor is absent from the figure because it was not measured.",
        "",
        "| Family | Band | 1D prior | P128n128 | Realization | POD | Ratio |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for family, label in (
        ("iid", "IID"),
        ("ood_dipping", "Dipping"),
        ("ood_three_layer", "Three-layer"),
        ("manifold_blind", "Manifold blind"),
        ("treasure_island", "Treasure Island"),
    ):
        for band in ("low", "mid", "high", "overall"):
            row = one(budget, family=family, band=band)
            ratio = "omitted" if pd.isna(row.ratio) else num(row.ratio, 3)
            real = "—" if pd.isna(row.realization_floor) else num(row.realization_floor)
            lines.append(
                f"| {label} | {band} | {num(row.prior_g_log_l1)} | {num(row.model_g_log_l1)} | {real} | {num(row.pod_floor)} | {ratio} |"
            )
    lines += [
        "",
        "Treasure Island rows in that table use the pooled mean. The interval stored beside them is a moving-block median and is not an interval for the mean (`error_budget.provenance.json`).",
        "",
        "## Consistency",
        "",
        "These tests are results. A failure is reported here, not moved to a footnote. Cutoffs are the proposals in `gates.py`: homogeneous median L1(|g_pred|) against "
        f"{num(floor, 4)}, mirror station-matched L1 against 0.1 times the median L1(g_true), monotone sign agreement 0.80.",
        "",
        "| Test | Band | Value | Reference | Result |",
        "| --- | --- | --- | --- | --- |",
    ]
    for test in ("homogeneous_column", "cov0", "mirror", "monotone_pairs"):
        part = consistency[consistency["test"] == test]
        for _, row in part.iterrows():
            band = row.band if isinstance(row.band, str) else "—"
            value = "—" if pd.isna(row.value) else num(row.value)
            reference = "—" if pd.isna(row.reference) else num(row.reference, 6)
            if row.status == "not_run":
                result = "not run"
            else:
                result = "pass" if bool(row.passed) else "fail"
            lines.append(f"| {test} | {band} | {value} | {reference} | {result} |")
    homo_high = one(consistency, test="homogeneous_column", band="high")
    lines += [
        "",
        f"The homogeneous column passes 0.1–0.4 Hz and 0.4–2.5 Hz and fails 2.5–10 Hz: median L1(|g_pred|) is {num(homo_high.value)} against the {num(floor, 4)} proposal. It also fails on 0.1–10 Hz. CoV = 0, built by repeating the center column of every IID test record, is worse than g = 0 on every band: g = 0 has log-L1 0 on that repeated column, and the network residual is the value in the table. The mirror test exceeds 0.1 times L1(g) on every band. Monotone pairs were not run: no existing pair changes only CoV or only dip, and a new finite-element pair was not solved.",
        "",
        "g = 0 remains the identity of the residual form. The CoV = 0 rows are the measurement that the network does not enact that identity on a repeated column.",
        "",
        "## Epoch 46",
        "",
        f"Epoch {int(epoch_row.epoch)} is the stored `best_model.pt`. `val_score` is {num(epoch_row.val_score, 5)}. The selection that stored it averages IID and manifold validation log-L1 (`select_families` {epoch_row.select_families}), query dtype {epoch_row.query_dtype}. Blind was not in the score. `history.csv` is absent from the checkpoint, so this page does not say epoch 46 is the minimum of a curve.",
        "",
        "Epochs 43, 44, 45, 47, 48, and 49 are not on disk and were not trained for this comparison. The four-checkpoint mean was not recomputed. Stored center-station curves exist for IID and dipping only. Those center-station pooled means are a different station set from the all-station ladder, and they are not a validation horse-race among the four checkpoints.",
        "",
        "| Family | Curve | Band | Center-station pooled g log-L1 |",
        "| --- | --- | --- | --- |",
    ]
    stored = epoch[epoch["status"] == "scored"]
    for _, row in stored.iterrows():
        if row.band not in ("mid", "high"):
            continue
        lines.append(
            f"| {row.family} | {row.method} | {row.band} | {num(row.g_log_l1)} |"
        )
    lines += [
        "",
        f"The all-station IID mid-band ladder value is {num(iid_mid.g_log_l1)}. The IID center-station row above is index 10 only. The two reductions are not averaged.",
        "",
        "## Treasure Island",
        "",
        "Treasure Island is one 4000 m section, `full_qvs.h5`, a 15 s Q(Vs) record. Each scored station is the center of a 2000 m window. The forward used for the ladder was not rerun in this revision; g is the stored `treasure_g.npz` named in the provenance sidecar. "
        f"The FFT window on that scoring is {treasure_prov['fft_window_s']} s. The bootstrap unit is `{treasure_prov['bootstrap_unit']}`. The interval in `ladder_treasure.csv` is the median of station g log-L1. The pooled mean is a separate column. Those two reductions are not combined into one interval.",
        "",
        f"Blind selection: {epoch_row.note} Treasure Island was not a selection input.",
        "",
        f"On the Phase 1 distance audit, Treasure Island's nearest-neighbor distance to the IID training cloud is {num(distance.nn_distance_to_iid_train, 3)}, percentile {num(distance.percentile_among_synthetic_test, 2)} among {int(distance.n_test)} synthetic test points (`splits.csv`). It sits at the far edge of those distances.",
        "",
        "Pooled g log-L1 on the 200 stations:",
        "",
        "| Band | P128n128 | g = 0 | Status |",
        "| --- | --- | --- | --- |",
    ]
    for band, label in (("low", "0.1–0.4 Hz"), ("mid", "0.4–2.5 Hz"), ("high", "2.5–10 Hz"), ("overall", "0.1–10 Hz")):
        model = one(treasure, rung="treasure_island", band=band)
        prior = one(baselines, rung="treasure_island", band=band, baseline="g0")
        lines.append(f"| {label} | {num(model.g_log_l1)} | {num(prior.g_log_l1)} | {model.accuracy_claim} |")
    ti_low = one(treasure, rung="treasure_island", band="low")
    lines += [
        "",
        f"Below 0.4 Hz the pooled model error is {num(ti_low.g_log_l1)} against a prior of {num(one(baselines, rung='treasure_island', band='low', baseline='g0').g_log_l1)}. The 128-mode basis keeps a fraction {num(ti_pod.fraction_kept, 4)} of that low-band residual (`representation.csv`). Halving the 15 s record moves the band by {leakage['low']['full_qvs.h5@7.490000000223517']:.3f}. The mismatch is strongly visible on the figure. The 15 s record does not validate that band, so it is not an accuracy result.",
        "",
        (
            f"On the scored mid band the median of the {int(one(identity, name='median_station_skill').n_stations)} stationwise skills is "
            f"{num(one(identity, name='median_station_skill').value, 4)}. "
            f"One minus the ratio of the separate median log-L1 values, "
            f"{num(one(identity, name='median_station_g_log_l1').value, 4)} and "
            f"{num(one(identity, name='median_station_prior_log_l1').value, 4)}, is "
            f"{num(one(identity, name='one_minus_ratio_of_medians').value, 4)}. "
            f"The skill of the mean errors is {num(one(identity, name='skill_of_mean_errors').value, 4)}. "
            "The median skill is not that ratio."
        ),
        "",
        "On the scored bands the prior in the table above exceeds the model. Both differences are smaller than the 15 s record-length leakage on those bands "
        f"({leakage['mid']['full_qvs.h5@7.490000000223517']:.3f} and {leakage['high']['full_qvs.h5@7.490000000223517']:.3f}).",
        "",
        "`05_treasure_island_full/central.pdf` marks 0.4 Hz and shows g_pred − g_true from the stored fields. Hatched bars in panel (d) are the low band and the overall band, which includes it. The center-column and Pretell outcrops for this section are in `baselines.csv`; they are a different column from the stored prior and are not drawn as an accuracy comparison.",
        "",
        "## Dropped",
        "",
        "Round-0 unpadded 2000 m sections include the absorbing boundary. `ladder.csv` records them as `not_scored_unpadded`. They are not clean 2000 m domains.",
        "",
        "`full_svlab.h5` matches `full_qvs.h5` surface motion. The ladder records it as `not_a_second_solve`. It was not scored, and it is not a damping experiment.",
        "",
        "G0 names a further difference and this page does not rebuild it: an earlier stage sentence chained a mean of per-station skills to the skill of a pooled error, and a three-layer stage bar was a mean of per-station skills clipped on the plot. Those reductions are omitted. The figure in `06_reliability_and_broadband/` is the pooled g log-L1 only.",
        "",
        "## What a wide-domain claim still needs",
        "",
        (
            "Current machines cannot run padded 2000 m widths, and an unpaired smaller solve would not answer the question. The missing labels are paired master geologies at "
            f"{_cited(rationale, 'future_width_m_500')}, {_cited(rationale, 'future_width_m_1000')}, "
            f"{_cited(rationale, 'future_width_m_1500')}, and {_cited(rationale, 'future_width_m_2000')} m, with "
            f"{_cited(rationale, 'future_pad_m')} m repeated-edge pads, depth through {_cited(rationale, 'future_depth_m')} m, "
            f"records of {_cited(rationale, 'future_record_s_short')}–{_cited(rationale, 'future_record_s_long')} s, "
            "gradient and layered Vs families, a CoV = 0 slice, and damping varied independently of Vs. "
            "A positive variable-domain claim needs those labels and a model trained from scratch in `experiments/GIFNO-VARDOMAIN/` "
            "with checkpoints in `checkpoints/vardomain/`. The present checkpoint is not that validation."
        ),
        "",
        "## Where the figures are",
        "",
        "- `01_method_overview/` — the column is kept, and g = 0 is an identity of the residual form.",
        f"- `02_tf_heatmaps_and_skill/` — stored 500 m exemplars, dipping run {_cited(exemplars, 'dipping_run')} and blind section man_0297. The caption states the selection rule. The fields were not redrawn.",
        "- `03_state_of_practice/` — elastic pooled g log-L1 against g = 0, the center column, and the 9-column Pretell mean. The mixed-prior practice box is not this figure.",
        "- `04_width_and_manifold/` — context invariance, the Case 7 column diagnostic, and the audit coverage figure.",
        "- `05_treasure_island_full/` — 0.4 Hz marked, g_pred − g_true shown. The low band is not an accuracy claim.",
        "- `06_reliability_and_broadband/` — pooled absolute g log-L1, validation beside blind, the error budget, and the consistency table above.",
        "",
        "The appendix states the epoch rule, the stored center curves, and the two files that were not scored. It does not add a second headline.",
        "",
    ]
    return lines


def _sections(ladder, treasure, baselines, consistency, epoch, splits, noise, floor, iid_mid, treasure_prov, epoch_prov, exemplars, identity) -> None:
    epoch_row = one(splits, table="epoch_rule")
    (HERE / "01_method_overview" / "README.md").write_text(
        "The window is the ground the operator sees. The column is the elastic outcrop under the station.\n\n"
        "The network returns g in |TF| = |TF_column| e^g. Setting g to zero returns the column. "
        "That is an identity of the residual parameterization, not a guarantee that a repeated column produces g = 0. "
        "The CoV = 0 rows in the consistency table are the check.\n\n"
        f"The checkpoint is P128n128, epoch {int(epoch_row.epoch)}, float32.\n"
    )
    dipping_run = _cited(exemplars, "dipping_run")
    (HERE / "02_tf_heatmaps_and_skill" / "README.md").write_text(
        f"Both stored figures are 500 m windows, and both are P128n128. "
        f"The dipping panel is dipping test run {dipping_run}, the held-out record with the largest "
        "center-station mid-band column residual. "
        "The realistic panel is blind section man_0297, the blind section with the largest stored mid-band |g| "
        "in the ensemble dump. Both are high-signal exemplars. They are not representative records and not wide-domain evidence.\n\n"
        "The spatial fields were not redrawn, because a new forward would be required. "
        "The caption that printed a median skill and a Pearson of linear |TF| was removed. "
        "Those reductions are not the pooled ladder. Pooled scores are in `analysis/frozen/ladder.csv`.\n\n"
        "The selection rule is also in `selection_rule.txt`.\n"
    )
    (HERE / "03_state_of_practice" / "README.md").write_text(
        "The bars are pooled g log-L1 on the elastic Q(Vs) Rayleigh outcrop: P128n128, g = 0, the center column, and the 9-column Pretell geometric mean. "
        "They are read from `analysis/frozen/ladder.csv` and `baselines.csv`. The plotted rows are copied to `plotted_from_frozen.csv`.\n\n"
        "Practice Pearson is of log|TF|. The previous box in this folder used Pearson of linear |TF| on the full spectrum (G0). "
        "That box is not the comparison, and this page does not reprint it.\n"
    )
    (HERE / "04_width_and_manifold" / "README.md").write_text(
        "Context invariance of pooled g log-L1. The training strip is 500 m. A wider window repeats the edge column. "
        "The curves are the `context_invariance` rows of `analysis/frozen/consistency_tests.csv`.\n\n"
        "`case7_column.pdf` is OpenSees versus the corrected column on the existing Case 7 H5s. "
        "The network was not scored. Open markers meet the domain edge of the 4000 m section. "
        "Those 2000 m windows are not a padded domain.\n\n"
        "The previous mid-band skill curve is not this figure.\n\n"
        "`coverage.png` is the Phase 1 coverage figure. It is copied from the audit and was not recomputed.\n"
    )
    ti_low = one(treasure, rung="treasure_island", band="low")
    ti_mid = one(treasure, rung="treasure_island", band="mid")
    (HERE / "05_treasure_island_full" / "README.md").write_text(
        "One 4000 m section. Each station is a 2000 m window. g_pred is the stored `treasure_g.npz` named in "
        f"`ladder_treasure.provenance.json` (FFT window {treasure_prov['fft_window_s']} s). The forward was not rerun.\n\n"
        "Panels (b) and (c) mark 0.4 Hz. Panel (c) is g_pred − g_true. "
        f"Panel (d) is pooled g log-L1. The low band is {num(ti_low.g_log_l1)} and is not an accuracy result. "
        f"The mid-band pooled value is {num(ti_mid.g_log_l1)}.\n\n"
        f"The median of the stationwise mid-band skills is {num(one(identity, name='median_station_skill').value, 4)}. "
        f"One minus the ratio of the median log-L1 values is {num(one(identity, name='one_minus_ratio_of_medians').value, 4)}. "
        "Those reductions are not the same.\n\n"
        "The per-station log-L1 curve from the previous figure was not redrawn. "
        "The moving-block interval is a median and is not drawn on the pooled mean.\n"
    )
    val = one(ladder, rung="manifold_val", band="mid")
    blind = one(ladder, rung="manifold_blind", band="mid")
    (HERE / "06_reliability_and_broadband" / "README.md").write_text(
        "Panel (a) is pooled g log-L1 on the in-family tests, mid band and high band, with the master-bootstrap interval. "
        "Panel (b) is the same reduction on manifold validation and manifold blind. "
        f"Mid-band validation is {num(val.g_log_l1)}; mid-band blind is {num(blind.g_log_l1)}. "
        "Blind was not in the selection score.\n\n"
        "There is no stage-skill bar on this figure.\n\n"
        "`error_budget.pdf` is drawn from `analysis/frozen/error_budget.csv`. The FE floor is unmeasured, and the ratios are provisional.\n\n"
        "The consistency table is in the main narrative. Homogeneous median L1 passes 0.1–2.5 Hz and fails 2.5–10 Hz. "
        "CoV = 0 is worse than g = 0 on every band. Mirror station-matched L1 exceeds 0.1 times L1(g) on every band. "
        "Monotone pairs were not run.\n"
    )
    (HERE / "appendix" / "learning_curve" / "README.md").write_text(
        f"Epoch {int(epoch_row.epoch)} is the stored checkpoint. `val_score` is {num(epoch_row.val_score, 5)}, "
        f"the mean of IID and manifold validation log-L1. Blind was not in the score. "
        f"{epoch_prov['definition_difference']}\n\n"
        "The old `history.csv` is in `superseded/` and is not the checkpoint history. It is not used for a minimum, and its losses are not quoted.\n"
    )
    (HERE / "appendix" / "architecture" / "README.md").write_text(
        "P128n128, epoch 46. Depth is a frozen code per column. Lateral layers mix a long Fourier path with a short path. "
        "The station query rebuilds g, and the caller adds the column.\n\n"
        "Loss, width, and parameter counts from earlier notes are not restated here. The scored model is the float32 checkpoint named in the ladder provenance.\n"
    )
    center = epoch[(epoch["status"] == "scored") & (epoch["band"] == "mid") & (epoch["family"] == "iid")]
    bits = []
    for _, row in center.iterrows():
        bits.append(f"{row.method} {num(row.g_log_l1)}")
    (HERE / "appendix" / "four_model_mean" / "README.md").write_text(
        "The four-checkpoint mean was not recomputed. Stored center-station curves exist for IID and dipping. "
        "They are not a best-of-four ranking on validation, and they are not the all-station ladder.\n\n"
        f"IID mid-band center-station pooled g log-L1: {', '.join(bits)}.\n\n"
        "The previous skill-versus-width figure for these checkpoints is not copied forward.\n"
    )
    (HERE / "07_one_model" / "README.md").write_text(
        "Scoring dumps from the previous draft stay in this folder. The narrative does not read them. "
        "The all-station pooled scores are `analysis/frozen/ladder.csv`.\n"
    )
    appendix = HERE / "appendix" / "damping_and_direct_ablation"
    appendix.mkdir(parents=True, exist_ok=True)
    (appendix / "README.md").write_text(
        "The previous direct-prediction table and the cached rigid-base sensitivity table are not inputs to this narrative. "
        "The comparison that is scored is the elastic pooled g log-L1 in `03_state_of_practice/`.\n\n"
        "G0: the legacy Pretell, Passeri, Dmult, and Toro curves are rigid-base arms. "
        "Harness g = 0 is the stored elastic column, which is a different prior.\n"
    )
    (HERE / "appendix" / "svlab_reference" / "README.md").write_text(
        "`full_svlab.h5` has the same surface motion as `full_qvs.h5`. It is not a second solve and was not scored. "
        "No Pearson is reported for it.\n"
    )
    # Silence unused-argument linters for tables the section pages do not repeat.
    _ = (baselines, consistency, noise, floor, iid_mid, exemplars)


if __name__ == "__main__":
    main()
