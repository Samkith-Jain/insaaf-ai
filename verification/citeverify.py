"""
CiteVerify (Statute-Precedent Matching Algorithm / SPMA).

Objective 4: verify every citation a downstream agent/explanation wants to
rely on, against a curated, authenticated source, before it's allowed into
the final IRAC output — minimizing hallucinated citations.

Two independent checks:
  1. verify_statute()   — section+act checked against the curated provisions
                           list (verification/curated_provisions.py).
  2. verify_precedent()  — case name checked against judgments actually
                           present in the structured legal DB, using
                           semantic textual similarity (difflib SequenceMatcher,
                           since sentence-transformers isn't installable in
                           this offline sandbox — see retrieval/dense/embed.py
                           for the same substitution) rather than exact match,
                           so paraphrased/partial citations still verify.

Each check returns a verdict + confidence, never a silent pass — an
unverifiable citation is surfaced as UNVERIFIED, not dropped silently,
so the explanation stage can flag it rather than hide it.
"""
import re
import difflib
from dataclasses import dataclass

from verification.curated_provisions import ACT_REGISTRY
from structuring.db import fetch_all_judgments


@dataclass
class CitationVerdict:
    citation_type: str  # "statute" | "precedent"
    citation_text: str
    verified: bool
    confidence: float
    matched_against: str | None = None
    reason: str = ""


def _normalize_act_name(act: str) -> str:
    return re.sub(r"\s+", " ", act.strip().lower()).rstrip(",")


def verify_statute(section: str, act: str) -> CitationVerdict:
    key = _normalize_act_name(act)
    citation_text = f"Section {section} of the {act}"

    provisions = ACT_REGISTRY.get(key)
    if provisions is None:
        # try fuzzy match on act name (typos, minor phrasing differences)
        candidates = difflib.get_close_matches(key, ACT_REGISTRY.keys(), n=1, cutoff=0.75)
        provisions = ACT_REGISTRY.get(candidates[0]) if candidates else None
        if provisions is None:
            return CitationVerdict("statute", citation_text, False, 0.0,
                                    reason=f"'{act}' not found in curated provisions database")

    if section in provisions:
        return CitationVerdict("statute", citation_text, True, 1.0,
                                matched_against=f"Section {section}: {provisions[section]}",
                                reason="exact section match in curated provisions")

    return CitationVerdict("statute", citation_text, False, 0.0,
                            reason=f"Section {section} not found under {act} in curated provisions database")


def verify_precedent(case_name: str, conn, similarity_threshold: float = 0.55) -> CitationVerdict:
    judgments = fetch_all_judgments(conn)
    best_score, best_match = 0.0, None

    target = case_name.lower().strip()
    for j in judgments:
        # match against case_number and against party names appearing in full_text header
        candidates = [j.get("case_number") or ""]
        header = (j.get("full_text") or "")[:400].lower()
        score_case_no = difflib.SequenceMatcher(None, target, (j.get("case_number") or "").lower()).ratio()
        score_header = max(
            (difflib.SequenceMatcher(None, target, line.strip()).ratio() for line in header.split("\n") if line.strip()),
            default=0.0,
        )
        score = max(score_case_no, score_header)
        if score > best_score:
            best_score, best_match = score, j

    if best_match and best_score >= similarity_threshold:
        return CitationVerdict("precedent", case_name, True, round(best_score, 3),
                                matched_against=best_match["case_number"],
                                reason="matched against a judgment present in the structured legal DB")

    return CitationVerdict("precedent", case_name, False, round(best_score, 3),
                            reason="no sufficiently similar judgment found in the curated legal DB "
                                   "(cited case may be genuine but outside this system's ingested corpus)")


def verify_all(sj, conn) -> list[CitationVerdict]:
    """Run CiteVerify over every citation extracted from a StructuredJudgment."""
    verdicts = []
    for s in sj.statute_citations:
        verdicts.append(verify_statute(s["section"], s["act"]))
    for p in sj.precedent_citations:
        verdicts.append(verify_precedent(p["case_name"], conn))
    return verdicts
