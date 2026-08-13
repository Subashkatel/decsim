"""Keep research evidence complete and outside production source files."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = REPO_ROOT / "docs/architecture/evidence_catalog.md"
PHASE_A_MAP = REPO_ROOT / "tmp/validation/core_evidence/README.md"

EVIDENCE_ID_PATTERN = re.compile(r"EV-[A-Z0-9]+(?:-[A-Z0-9]+)+")
SOURCE_EVIDENCE_PATTERN = re.compile(
    r"(?:^|[\n#])\s*(?:Evidence|Grounding):\s*|tmp/validation/|"
    r"EV-[A-Z0-9]+(?:-[A-Z0-9]+)+|arXiv:|doi:",
    re.IGNORECASE,
)


def _load_catalog_registry():
    text = CATALOG_PATH.read_text(encoding="utf-8")
    quarantine_start = text.index("## QUARANTINED")
    registered = set(EVIDENCE_ID_PATTERN.findall(text[:quarantine_start]))
    quarantined = set(EVIDENCE_ID_PATTERN.findall(text[quarantine_start:]))
    return registered - quarantined, quarantined


def test_evidence_maps_and_classifications_exist():
    for path in (CATALOG_PATH, PHASE_A_MAP):
        assert path.exists(), f"missing evidence file: {path}"
    catalog = CATALOG_PATH.read_text(encoding="utf-8")
    for classification in (
        "DIRECT_QEC_CODE",
        "DIRECT_CLASSICAL_CODE",
        "PAPER_ONLY",
        "DECSIM_POLICY",
    ):
        assert classification in catalog


def test_external_phase_a_ids_are_registered_and_not_quarantined():
    registered, quarantined = _load_catalog_registry()
    cited = set(EVIDENCE_ID_PATTERN.findall(
        PHASE_A_MAP.read_text(encoding="utf-8")))
    assert cited, "Phase A evidence map cites no evidence IDs"
    assert not cited - registered, f"unknown or quarantined evidence IDs: {cited - registered}"
    assert "EV-QUARANTINE-ASTREA-PDF" in quarantined
    assert not any("ASTREA" in evidence_id for evidence_id in cited)


def test_production_python_contains_no_research_evidence_tags():
    offending = {}
    for path in (REPO_ROOT / "decsim").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        hits = sorted(set(match.group(0) for match in
                          SOURCE_EVIDENCE_PATTERN.finditer(text)))
        if hits:
            offending[str(path.relative_to(REPO_ROOT))] = hits
    assert not offending, (
        "research citations belong in tmp/validation evidence maps, not source: "
        f"{offending}"
    )
