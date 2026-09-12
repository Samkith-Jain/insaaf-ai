"""
Stage: IRAC Explanation Generation.

Objective 4 (second half): structure a judgment's verified citations and
text into Issue / Rule / Application / Conclusion format.

Production target (per architecture diagram): LLM-based NLG. This sandbox
has no network access to call an LLM API, so this is rule-based extractive
NLG — it locates the sentences most likely to express each IRAC component
using position + keyword heuristics (e.g. "seeking", "grievance" -> Issue;
"Section X of" -> Rule; final numbered directions -> Conclusion) rather than
generating novel prose. It's a real, inspectable extraction, not a
placeholder — but it will read as terser and less fluent than an LLM would
produce, and that's an honest tradeoff to state, not a hidden gap.
"""
import re
from dataclasses import dataclass, field

ISSUE_KEYWORDS = re.compile(r"\b(grievance|seeking|complaint|dispute|question|issue)\b", re.IGNORECASE)
CONCLUSION_MARKERS = re.compile(r"\b(disposed of with the following directions|it is (hereby )?(ordered|directed)|we (dismiss|allow)|complaint is (partly )?allowed|complaint is disposed)\b", re.IGNORECASE)
DIRECTION_LINE = re.compile(r"^\(?\s*[ivxIVX]+\s*[\)\.]\s+", re.MULTILINE)


@dataclass
class IRACExplanation:
    judgment_id: int | None
    case_number: str | None
    issue: str
    rule: list
    application: str
    conclusion: str
    verified_citation_count: int = 0
    unverified_citation_count: int = 0
    citation_verdicts: list = field(default_factory=list)


def _sentences(text: str) -> list[str]:
    # naive sentence split — good enough for extractive selection on legal prose
    return [s.strip() for s in re.split(r"(?<=[.;])\s+(?=[A-Z0-9(])", text) if len(s.strip()) > 20]


def _extract_issue(body_text: str) -> str:
    sents = _sentences(body_text)[:15]  # issue usually stated early
    matches = [s for s in sents if ISSUE_KEYWORDS.search(s)]
    if matches:
        return matches[0]
    return sents[0] if sents else "Issue not clearly extractable from available text."


def _extract_rule(statute_citations: list[dict], precedent_citations: list[dict]) -> list[str]:
    rule_lines = []
    for s in statute_citations:
        rule_lines.append(f"Section {s['section']} of the {s['act']}")
    for p in precedent_citations[:5]:  # cap to keep it readable
        rule_lines.append(p["case_name"] + (f" ({p['year']})" if p.get("year") else ""))
    return rule_lines


def _extract_application(body_text: str) -> str:
    sents = _sentences(body_text)
    # application = reasoning sentences, typically the bulk between issue and conclusion,
    # excluding lines that are pure directions/orders
    mid = [s for s in sents if not CONCLUSION_MARKERS.search(s) and not DIRECTION_LINE.match(s)]
    # take a representative middle slice
    if len(mid) <= 3:
        return " ".join(mid)
    start = len(mid) // 4
    return " ".join(mid[start:start + 3])


def _extract_conclusion(body_text: str) -> str:
    sents = _sentences(body_text)
    conclusion_sents = [s for s in sents if CONCLUSION_MARKERS.search(s)]
    if conclusion_sents:
        idx = sents.index(conclusion_sents[0])
        return " ".join(sents[idx:idx + 4])
    # fallback: last few sentences (orders are usually at the end)
    return " ".join(sents[-3:]) if sents else "Conclusion not clearly extractable."


def generate_irac(sj, judgment_id: int | None, citation_verdicts: list = None) -> IRACExplanation:
    citation_verdicts = citation_verdicts or []
    verified = sum(1 for v in citation_verdicts if v.verified)
    unverified = len(citation_verdicts) - verified

    return IRACExplanation(
        judgment_id=judgment_id,
        case_number=sj.case_number,
        issue=_extract_issue(sj.body_text or sj.full_text),
        rule=_extract_rule(sj.statute_citations, sj.precedent_citations),
        application=_extract_application(sj.body_text or sj.full_text),
        conclusion=_extract_conclusion(sj.body_text or sj.full_text),
        verified_citation_count=verified,
        unverified_citation_count=unverified,
        citation_verdicts=citation_verdicts,
    )


def render_irac(irac: IRACExplanation) -> str:
    lines = [
        f"=== IRAC Explanation — {irac.case_number or 'Unnumbered case'} ===",
        f"\nISSUE:\n{irac.issue}",
        f"\nRULE:\n" + ("\n".join(f"  - {r}" for r in irac.rule) if irac.rule else "  (no citations extracted)"),
        f"\nAPPLICATION:\n{irac.application}",
        f"\nCONCLUSION:\n{irac.conclusion}",
        f"\nCitation verification: {irac.verified_citation_count} verified, "
        f"{irac.unverified_citation_count} unverified",
    ]
    for v in irac.citation_verdicts:
        status = "✓ VERIFIED" if v.verified else "✗ UNVERIFIED"
        lines.append(f"  [{status}, conf={v.confidence}] ({v.citation_type}) {v.citation_text or v.matched_against} — {v.reason}")
    return "\n".join(lines)
