"""
Stage 2: Structuring.
- Segments a judgment into rough sections (header/parties, facts/body, order/directions).
- Extracts statute citations and case-law precedent citations via regex-based NER.
  (spaCy is listed in requirements.txt for production; this environment has no
  network access to pull model weights, so citation/entity extraction here is
  rule-based — legal citations follow tight, predictable patterns, which makes
  regex NER a reasonable and fast baseline ahead of swapping in a trained NER model.)
"""
import re
from dataclasses import dataclass, field


STATUTE_PATTERN = re.compile(
    r"Section\s+(\d+[A-Za-z]?(?:\(\d+\))?)\s+of\s+the\s+([A-Z][A-Za-z,\.\s]{3,80}?Act,?\s*\d{4})",
    re.IGNORECASE,
)

# "X Vs. Y ... (YEAR) N SCC N" or "X Vs. Y [YEAR] ... (SC)" or "CC No. N of YEAR"
CASE_CITATION_PATTERN = re.compile(
    r"([A-Z][A-Za-z.&\s]{2,70})\s+(?:Vs\.?|v\.?|Versus)\s+([A-Z][A-Za-z.&\s]{2,70}?)"
    r"(?=\s*[\[\(]?\d{4}[\]\)]?|\s*,|\s*\.\s+[A-Z]|\n|$)"
    r"(?:\s*[\[\(](\d{4})[\]\)])?"
    r"(?:\s*[,]?\s*(\d+\s*SCC\s*\d+|[IVX]+\s*\(\d{4}\)\s*CPJ\s*\d+\s*\(SC\)))?",
)

CASE_NUMBER_PATTERN = re.compile(
    r"(Consumer Case No\.?\s*\d+\s*of\s*\d{4}|CC No\.?\s*\d+\s*of\s*\d{4}|Appeal No\.?\s*[A-Z/\d]+)",
    re.IGNORECASE,
)

DATE_PATTERN = re.compile(r"Dated?\s*:?\s*(\d{1,2}[\.\-/\s]\w+[\.\-/\s]\d{4}|\d{1,2}\w{2}\s+\w+,?\s+\d{4})", re.IGNORECASE)

SECTION_HEADERS = {
    "order": re.compile(r"^(ORDER|ORAL ORDER|JUDGMENT)\s*$", re.IGNORECASE),
}


@dataclass
class StructuredJudgment:
    forum: str | None = None
    case_number: str | None = None
    date: str | None = None
    parties: dict = field(default_factory=dict)
    bench: list = field(default_factory=list)
    statute_citations: list = field(default_factory=list)
    precedent_citations: list = field(default_factory=list)
    header_text: str = ""
    body_text: str = ""
    full_text: str = ""


def detect_forum(text: str) -> str | None:
    if re.search(r"NATIONAL CONSUMER DISPUTES REDRESSAL COMMISSION", text, re.IGNORECASE):
        return "NCDRC"
    if re.search(r"STATE CONSUMER DISPUTES REDRESSAL COMMISSION", text, re.IGNORECASE):
        return "State Commission"
    if re.search(r"DISTRICT (CONSUMER )?(DISPUTES REDRESSAL )?(FORUM|COMMISSION)", text, re.IGNORECASE):
        return "District Forum"
    return None


def extract_statutes(text: str) -> list[dict]:
    seen = set()
    out = []
    for m in STATUTE_PATTERN.finditer(text):
        section, act = m.group(1).strip(), re.sub(r"\s+", " ", m.group(2)).strip().rstrip(",")
        key = (section, act.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append({"section": section, "act": act, "span": m.span()})
    return out


def extract_precedents(text: str) -> list[dict]:
    seen = set()
    out = []
    for m in CASE_CITATION_PATTERN.finditer(text):
        party_a, party_b = m.group(1).strip(), m.group(2).strip()
        # filter obvious false positives (too short / generic words)
        if len(party_a) < 3 or len(party_b) < 3:
            continue
        if party_a.lower() in ("vs", "versus") or party_b.lower() in ("vs", "versus"):
            continue
        year, reporter = m.group(3), m.group(4)
        key = (party_a.lower(), party_b.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "case_name": f"{party_a} Vs. {party_b}",
            "year": year,
            "reporter_citation": reporter,
            "span": m.span(),
        })
    return out


def segment(cleaned_text: str) -> StructuredJudgment:
    sj = StructuredJudgment(full_text=cleaned_text)
    sj.forum = detect_forum(cleaned_text)

    case_no_match = CASE_NUMBER_PATTERN.search(cleaned_text)
    sj.case_number = case_no_match.group(1) if case_no_match else None

    date_match = DATE_PATTERN.search(cleaned_text)
    sj.date = date_match.group(1) if date_match else None

    # split header (parties/bench) vs body/order at the ORDER heading
    lines = cleaned_text.split("\n")
    split_idx = len(lines)
    for i, line in enumerate(lines):
        if SECTION_HEADERS["order"].match(line.strip()):
            split_idx = i
            break
    sj.header_text = "\n".join(lines[:split_idx])
    sj.body_text = "\n".join(lines[split_idx:]) if split_idx < len(lines) else cleaned_text

    # parties: naive "Versus" split within header
    vs_match = re.search(r"(.+?)\s*\n?Versus\s*\n?(.+?)(?:\n\s*BEFORE|\Z)", sj.header_text, re.IGNORECASE | re.DOTALL)
    if vs_match:
        sj.parties = {
            "complainant_block": vs_match.group(1).strip()[-300:],
            "opposite_party_block": vs_match.group(2).strip()[:300],
        }

    bench_matches = re.findall(r"(?:HON'?BLE\s+)?(?:MR\.?|MRS\.?|MS\.?|JUSTICE|DR\.?)\s*[A-Z][A-Za-z\.\s]{2,40}?(?:,\s*(?:PRESIDENT|MEMBER|PRESIDING MEMBER))", cleaned_text)
    sj.bench = list(dict.fromkeys(b.strip() for b in bench_matches))[:5]

    sj.statute_citations = extract_statutes(cleaned_text)
    sj.precedent_citations = extract_precedents(cleaned_text)

    return sj


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from data_pipeline.extract import extract
    from data_pipeline.clean import clean_text

    result = extract(sys.argv[1])
    cleaned = clean_text(result["text"])
    sj = segment(cleaned)
    print("Forum:", sj.forum)
    print("Case No.:", sj.case_number)
    print("Date:", sj.date)
    print("Bench:", sj.bench)
    print("Statutes:", sj.statute_citations)
    print("Precedents:", sj.precedent_citations)
