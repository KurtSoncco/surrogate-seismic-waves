#!/usr/bin/env python3
"""Trace every numeric literal in the generated geotech_v2 READMEs.

Each number must match a row in an analysis/frozen or analysis/audit summary
table, and that table must have a provenance sidecar. Numbers that live only
under a superseded/ directory are not headlines. Years, epoch 46, the frozen
sample sizes 720, 90, 24, and 3, and the band edges 0.1, 0.4, 2.5, and 10 are
not required to match.

The same pass fails closed on the stale comparisons Phase 3 was written to
drop: the stage chain 0.616, the clipped bar -5.689, the SVLAB Pearson 0.881
as a second-solve claim, a practice box described as rigid-base, an unlabeled
dtype, and Case 7 "vs 1D" numbers from the old half-space engine.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent / "paper" / "geotech_v2"
FROZEN = HERE / "frozen"
AUDIT = HERE / "audit"

# Headline tables only. Station dumps and per-record coordinate tables are not
# sources: a rounded station index would hide an unsourced headline.
FROZEN_TABLES = (
    "ladder.csv",
    "ladder_treasure.csv",
    "baselines.csv",
    "error_budget.csv",
    "consistency_tests.csv",
    "epoch_neighborhood.csv",
    "case7_column.csv",
    "cache_engine_check.csv",
    "treasure_skill_identity.csv",
    "dtype_stamp.csv",
    "heatmap_exemplars.csv",
)
AUDIT_TABLES = (
    "census.csv",
    "splits.csv",
    "representation.csv",
    "signal.csv",
    "label_floors.csv",
    "noise_floor.json",
    "domain_rationale.csv",
)
# realization.csv is one row per Sobol group. Its group_id column would make
# any integer from that range look sourced. The floors quoted in the narrative
# are the error-budget and baseline rows.
IDENTIFIER_COLUMNS = {"order", "group_id", "run_index", "sample_id"}

NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"([+\-−]?(?:\d+\.\d+|\d+)(?:[eE][+\-−]?\d+)?)"
    r"(?![A-Za-z0-9_])"
)
PRECISION_RE = re.compile(
    r"\b(?:float32|float16|bfloat16|bf16|fp32|fp16|load_fp32)\b",
    re.IGNORECASE,
)
DTYPE_WORD_RE = re.compile(r"\bdtype\b", re.IGNORECASE)
ROLE_RE = re.compile(
    r"\b(?:checkpoint|query|forward|load_fp32|load|model|agreement|frozen|autocast|provenance|sidecar)\b",
    re.IGNORECASE,
)
VS_1D_RE = re.compile(r"\b(?:vs\.?|versus)\s+1\s*D\b", re.IGNORECASE)
CASE7_CONTEXT_RE = re.compile(
    r"\b(?:vs\.?|versus)\s+1\s*D\b|\bLOGLO\b|\bhalf-space engine\b|\bCase 7\b|\bthinning\b|\bflat west\b|\bflat east\b",
    re.IGNORECASE,
)

# Printed forms of the old half-space "vs 1D" comparisons. A token that also
# rounds out of a frozen or audit row is a current headline, not that claim.
CASE7_LEGACY = {
    "0.977",
    "0.983",
    "0.959",
    "0.965",
    "0.799",
    "0.864",
    "0.974",
    "0.984",
    "0.982",
    "0.226",
    "0.203",
    "0.374",
    "0.367",
    "0.952",
    "0.55",
    "-0.55",
}
SAMPLE_SIZES = {720, 90, 24, 3}
BAND_EDGE_DECIMALS = {0.1: 1, 0.4: 1, 2.5: 1, 10.0: 0}


@dataclass(frozen=True)
class Literal:
    path: Path
    line: int
    text: str
    value: float
    decimals: int
    percent: bool
    sentence: str

    @property
    def where(self) -> str:
        return f"{self.path.relative_to(PAPER)}:{self.line}: {self.text}"


@dataclass(frozen=True)
class SourceValue:
    value: float
    path: Path
    location: str


def generated_readmes() -> list[Path]:
    found = []
    for path in PAPER.rglob("README.md"):
        if "superseded" in path.parts:
            continue
        found.append(path)
    return sorted(found)


def decimal_places(token: str) -> int:
    body = token.lower().split("e", 1)[0]
    if "." not in body:
        return 0
    return len(body.split(".", 1)[1])


def parse_token(token: str) -> float:
    return float(token.replace("−", "-").replace("+", "", 1) if token.startswith("+") else token.replace("−", "-"))


def normalized_token(token: str) -> str:
    text = token.replace("−", "-")
    if text.startswith("+"):
        text = text[1:]
    return text


def ignored(token: str, value: float) -> bool:
    decimals = decimal_places(token)
    if "e" not in token.lower():
        for edge, places in BAND_EDGE_DECIMALS.items():
            if decimals == places and math.isclose(value, edge, rel_tol=0.0, abs_tol=1e-12):
                return True
        if decimals == 0 and math.isclose(value, round(value), rel_tol=0.0, abs_tol=1e-8):
            integer = int(round(value))
            if integer == 46 or integer in SAMPLE_SIZES:
                return True
            if 2010 <= integer <= 2035:
                return True
    return False


def sentence_at(text: str, start: int) -> str:
    left = max(text.rfind("\n", 0, start), text.rfind(".", 0, start), text.rfind("!", 0, start), text.rfind("?", 0, start))
    right_candidates = [text.find(mark, start) for mark in ("\n", ".", "!", "?")]
    right_candidates = [item for item in right_candidates if item != -1]
    right = min(right_candidates) if right_candidates else len(text)
    return text[left + 1 : right + 1].strip()


def literals_in(path: Path) -> list[Literal]:
    text = path.read_text()
    found = []
    for match in NUMBER_RE.finditer(text):
        raw = match.group(1)
        token = normalized_token(raw)
        try:
            value = parse_token(raw)
        except ValueError:
            continue
        if ignored(token, value):
            continue
        percent = text[match.end() : match.end() + 1] == "%"
        line = text.count("\n", 0, match.start()) + 1
        found.append(
            Literal(
                path=path,
                line=line,
                text=token + ("%" if percent else ""),
                value=value,
                decimals=decimal_places(token),
                percent=percent,
                sentence=sentence_at(text, match.start()),
            )
        )
    return found


def formats_as(source: float, literal: Literal) -> bool:
    token = literal.text[:-1] if literal.percent else literal.text
    if "e" in token.lower():
        return math.isclose(source, literal.value, rel_tol=1e-12, abs_tol=0.0)
    if literal.decimals == 0:
        if abs(source - round(source)) > 1e-6:
            return False
        rendered = str(int(round(source)))
        return rendered == token
    if f"{source:.{literal.decimals}f}" == token:
        return True
    if literal.percent and 0.0 <= source <= 1.0:
        return f"{100.0 * source:.{literal.decimals}f}" == token
    return False


def add_number(bucket: list[SourceValue], value: float, path: Path, location: str) -> None:
    if math.isnan(value) or math.isinf(value):
        return
    bucket.append(SourceValue(value, path, location))


def numbers_in_text(text: str) -> list[float]:
    stripped = text.strip()
    if stripped.lower() in {"true", "false", "nan", "none", "inf", "-inf", "+inf"}:
        return []
    try:
        return [float(stripped)]
    except ValueError:
        found = []
        for match in NUMBER_RE.finditer(text):
            try:
                found.append(parse_token(match.group(1)))
            except ValueError:
                continue
        return found


def walk_json(node, path: Path, location: str, bucket: list[SourceValue]) -> None:
    if isinstance(node, bool) or node is None:
        return
    if isinstance(node, (int, float)):
        add_number(bucket, float(node), path, location)
        return
    if isinstance(node, str):
        for value in numbers_in_text(node):
            add_number(bucket, value, path, location)
        return
    if isinstance(node, list):
        for index, item in enumerate(node):
            walk_json(item, path, f"{location}[{index}]", bucket)
        return
    if isinstance(node, dict):
        for key, item in node.items():
            walk_json(item, path, f"{location}.{key}", bucket)


def load_table(path: Path) -> list[SourceValue]:
    bucket: list[SourceValue] = []
    if path.suffix == ".json":
        walk_json(json.loads(path.read_text()), path, path.name, bucket)
        return bucket
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        skip = {index for index, name in enumerate(header) if name in IDENTIFIER_COLUMNS}
        for row_index, row in enumerate(reader, start=2):
            for column, cell in enumerate(row):
                if column in skip or cell == "":
                    continue
                for value in numbers_in_text(cell):
                    add_number(bucket, value, path, f"{path.name}:{row_index}:{column}")
    return bucket


def provenance_numbers(directory: Path) -> list[float]:
    found: list[float] = []
    if not directory.is_dir():
        return found
    for path in directory.glob("*.provenance.json"):
        walk_json(json.loads(path.read_text()), path, path.name, found_holder := [])
        found.extend(item.value for item in found_holder)
    return found


def has_sidecar(source: SourceValue, sidecar_numbers: dict[Path, list[float]]) -> bool:
    sidecar = source.path.with_name(source.path.stem + ".provenance.json")
    if sidecar.is_file():
        return True
    for number in sidecar_numbers.get(source.path.parent, []):
        if math.isclose(number, source.value, rel_tol=1e-12, abs_tol=1e-15):
            return True
    return False


def load_sources() -> list[SourceValue]:
    sources: list[SourceValue] = []
    missing = []
    for name in FROZEN_TABLES:
        path = FROZEN / name
        if not path.is_file():
            missing.append(path)
            continue
        sources.extend(load_table(path))
    for name in AUDIT_TABLES:
        path = AUDIT / name
        if not path.is_file():
            missing.append(path)
            continue
        sources.extend(load_table(path))
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise SystemExit(f"missing source table: {joined}")
    return sources


def match_literal(literal: Literal, sources: list[SourceValue], sidecar_numbers: dict[Path, list[float]]) -> SourceValue | None:
    for source in sources:
        if not formats_as(source.value, literal):
            continue
        if has_sidecar(source, sidecar_numbers):
            return source
    return None


def stale_hits(path: Path, literals: list[Literal], matched: set[int]) -> list[str]:
    text = path.read_text()
    hits = []
    if "0.616" in text:
        hits.append(f"stale {path.relative_to(PAPER)}: stage chain 0.616")
    if "-5.689" in text or "−5.689" in text:
        hits.append(f"stale {path.relative_to(PAPER)}: clipped stage bar -5.689")
    if "0.881" in text:
        hits.append(f"stale {path.relative_to(PAPER)}: SVLAB Pearson 0.881 as a second-solve claim")
    if VS_1D_RE.search(text):
        hits.append(f"stale {path.relative_to(PAPER)}: Case 7 legacy vs 1D claim")
    for literal in literals:
        token = literal.text[:-1] if literal.percent else literal.text
        if token not in CASE7_LEGACY:
            continue
        if id(literal) in matched and not CASE7_CONTEXT_RE.search(literal.sentence):
            continue
        hits.append(f"stale {literal.where}: Case 7 legacy vs 1D number {token}")
    for sentence in re.split(r"\n+|(?<=[.!?])\s+", text):
        if not sentence.strip():
            continue
        lowered = sentence.lower()
        if "rigid-base" not in lowered and "rigid base" not in lowered:
            dtype_problem = _unlabeled_dtype(sentence)
            if dtype_problem:
                line = _line_of(text, sentence)
                hits.append(f"stale {path.relative_to(PAPER)}:{line}: unlabeled dtype ({dtype_problem})")
            continue
        negated = any(
            phrase in lowered
            for phrase in (
                "not a rigid",
                "not the comparison",
                "are not",
                "is not",
                "not inputs",
                "not the rigid",
            )
        )
        names_box = "rigid-base box" in lowered or "rigid base box" in lowered
        describes_practice = any(
            phrase in lowered
            for phrase in ("practice box", "the box", "in-family comparison", "state of practice", "bars are")
        )
        if names_box and not negated:
            line = _line_of(text, sentence)
            hits.append(f"stale {path.relative_to(PAPER)}:{line}: rigid-base describes the practice box")
        elif describes_practice and not negated:
            line = _line_of(text, sentence)
            hits.append(f"stale {path.relative_to(PAPER)}:{line}: rigid-base describes the practice box")
        dtype_problem = _unlabeled_dtype(sentence)
        if dtype_problem:
            line = _line_of(text, sentence)
            hits.append(f"stale {path.relative_to(PAPER)}:{line}: unlabeled dtype ({dtype_problem})")
    return hits


def _unlabeled_dtype(sentence: str) -> str:
    has_word = DTYPE_WORD_RE.search(sentence) is not None
    has_precision = PRECISION_RE.search(sentence) is not None
    has_role = ROLE_RE.search(sentence) is not None
    if has_word and not has_precision:
        return "dtype without float32, bf16, or load_fp32"
    if has_precision and not has_role:
        return "precision without checkpoint, query, forward, or model"
    return ""


def _line_of(text: str, sentence: str) -> int:
    start = text.find(sentence.strip())
    if start < 0:
        return 1
    return text.count("\n", 0, start) + 1


def main() -> int:
    sources = load_sources()
    sidecar_numbers = {
        FROZEN: provenance_numbers(FROZEN),
        AUDIT: provenance_numbers(AUDIT),
    }
    pages = generated_readmes()
    if not pages:
        print("fail")
        print("orphan: no generated README.md under paper/geotech_v2")
        return 1
    orphans: list[str] = []
    stale: list[str] = []
    checked = 0
    for path in pages:
        page_literals = literals_in(path)
        matched: set[int] = set()
        for literal in page_literals:
            checked += 1
            source = match_literal(literal, sources, sidecar_numbers)
            if source is None:
                orphans.append(f"orphan {literal.where}")
            else:
                matched.add(id(literal))
        stale.extend(stale_hits(path, page_literals, matched))
    if orphans or stale:
        print("fail")
        for line in orphans + stale:
            print(line)
        print(f"checked {checked} numeric literals in {len(pages)} READMEs")
        return 1
    print("pass")
    print(f"checked {checked} numeric literals in {len(pages)} READMEs, 0 orphans")
    return 0


if __name__ == "__main__":
    sys.exit(main())
