"""
Agent 1 of the Multi-Agent Legal Reasoning Pipeline: FACT EXTRACTION AGENT.

Objective (project synopsis): use text classification + argument mining to
extract case facts, so that the downstream agents (Statute Retrieval,
Precedent Retrieval, Argument Analysis, Prediction) reason over a structured,
auditable fact sheet instead of raw text.

What it does
------------
1. Isolates the narrative body of the case file (skips the party/bench header).
2. Optionally removes the operative order ("disposed of with the following
   directions ...") so that the Prediction Agent is never handed the answer.
   This is *outcome-leakage control*: it matters whenever the input is a
   decided judgment, which is exactly how the system is evaluated.
3. Splits the body into sentences that keep exact character offsets.
4. Classifies every sentence (argument mining) as one of:
       transaction | payment | deficiency | claim | defence | procedural |
       legal_reasoning | other
   `transaction`, `payment`, `deficiency` are FACTS; `claim` / `defence` are
   the two sides' positions (feed Argument Analysis); `procedural` holds
   limitation / jurisdiction points; `legal_reasoning` is the court's own law
   discussion and is kept OUT of the fact list.
5. Extracts entities: parties, dates, money, durations, percentages.
6. Builds a normalised timeline and *derived* facts (e.g. promised
   possession deadline and delay), each tagged derived=True with its inputs.
7. Detects the dispute category and CPA-2019 concept hooks
   (deficiency in service, defect in goods, unfair trade practice).
8. Emits a retrieval query + hooks for the Statute Retrieval Agent.

Transparency
------------
Every fact carries its paragraph number and (start, end) offsets into the
input, and `grounded` is verified by re-slicing the source. Nothing is
generated - a fact is either a verbatim sentence of the case file or it does
not exist.

Rules vs. LLM
-------------
Default is an offline rule-based classifier (same trade-off documented in the
README for NER / IRAC). Pass `llm=AnthropicBackend()` (agents/llm.py) to have
an LLM re-label the sentences; on any failure it falls back to the rules.
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field, asdict
from datetime import date

from agents.base import BaseAgent

# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------
FACT_LABELS = ("transaction", "payment", "deficiency")
ALL_LABELS = FACT_LABELS + ("claim", "defence", "procedural", "legal_reasoning", "other")

# --------------------------------------------------------------------------
# Entity patterns
# --------------------------------------------------------------------------
MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}
MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_name) if m})

DATE_NUMERIC = re.compile(r"(?<![\d.])(\d{1,2})[./-](\d{1,2})[./-](\d{4})(?![\d])")
DATE_WORDY = re.compile(
    r"(?<!\d)(\d{1,2})(?:st|nd|rd|th)?\s+(" + "|".join(sorted(MONTHS, key=len, reverse=True)) +
    r")\.?,?\s+(\d{4})",
    re.IGNORECASE,
)
MONEY = re.compile(
    r"(?:Rs\.?|INR|₹)\s*([\d,]+(?:\.\d+)?)\s*(?:/-)?\s*(lakhs?|lacs?|crores?)?",
    re.IGNORECASE,
)
PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|per\s*cent)", re.IGNORECASE)
NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "eighteen": 18, "twenty-four": 24, "thirty-six": 36,
}
DURATION = re.compile(
    r"\b(\d+\s+\d/\d|\d+(?:\.\d+)?|" + "|".join(NUMBER_WORDS) + r")\s*(days?|months?|years?)\b",
    re.IGNORECASE,
)
UNIT_MULT = {"lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "crore": 1e7, "crores": 1e7}

# --------------------------------------------------------------------------
# Sentence-classification rules (argument mining, rule-based baseline)
# Each label -> list of (regex, weight). Highest total wins.
# --------------------------------------------------------------------------
def _rx(p):
    return re.compile(p, re.IGNORECASE)


RULES: dict[str, list[tuple[re.Pattern, float]]] = {
    "legal_reasoning": [
        (_rx(r"\b(reference in this regard|can be made to the decision|as held by|this (commission|court) held)\b"), 3),
        (_rx(r"\b(hon'?ble (supreme court|high court)|apex court|bench of this commission)\b"), 2),
        (_rx(r"\b(vs\.?|versus)\b.*\b(\(\d{4}\)|\[\d{4}\]|SCC|CPJ|CC No)"), 2),
        (_rx(r"\bin terms of section\b|\bas defined by section\b|\bwould mean\b|\bentitled to (seek|claim)\b"), 2),
        (_rx(r"\b(amounts? to|clearly amounts)\b.*\b(deficiency|unfair)\b"), 8),
        (_rx(r"\b(cannot be compelled|cannot be made to wait|recurrent cause of action)\b"), 2),
    ],
    "defence": [
        (_rx(r"\b(resisted|contested|controverted)\b"), 3),
        (_rx(r"\b(o\.?p\.?|opposite part(y|ies)|developer|builder|respondent|insurer|bank)\b.{0,40}\b(contend|submit|plead|denied|deny|objection|taken|took|stated)"), 3),
        (_rx(r"\bpreliminary objection|written (version|statement)|reply filed\b"), 3),
        (_rx(r"\b(force majeure|not liable|no deficiency|barred by)\b"), 2),
    ],
    "claim": [
        (_rx(r"\b(grievance of the complainants?|complainants? (is|are)\s+(therefore,?\s+)?before)\b"), 3),
        (_rx(r"\b(seeking|prayed|praying|prays)\b"), 3),
        (_rx(r"\b(claims?|demand(ed|ing)?)\b"), 1),
        (_rx(r"\b(refund|compensation|interest|damages|replacement|litigation costs?)\b.{0,60}\b(seeking|prayed|claim)"), 2),
        (_rx(r"\bpreferred (this|the) (complaint|appeal)\b|\bfiled (this|the|a) (consumer )?complaint\b"), 2),
    ],
    "procedural": [
        (_rx(r"\b(limitation|pecuniary jurisdiction|territorial jurisdiction|maintainab|locus standi|condon)"), 4),
        (_rx(r"\b(legal notice|notice dated|notice was (sent|issued)|served)\b"), 2),
        (_rx(r"\b(ex[- ]?parte|adjourn|hearing)\b"), 1),
    ],
    "deficiency": [
        (_rx(r"\b(delay(ed)?|not (yet )?(complete|completed|delivered|offered|handed|refunded|paid|resolved|replaced)|failed to|failure to|fail(ed|s) to)\b"), 3),
        (_rx(r"\b(defect(ive|s)?|deficien\w+|negligen\w+|repudiat\w+|overcharg\w+|misrepresent\w+|unfair trade|not as (promised|described))\b"), 3),
        (_rx(r"\b(till date|to date|even now|despite (repeated )?(requests?|reminders?|follow[- ]?ups?))\b"), 2),
        (_rx(r"\b(no (possession|response|reply|refund)|did not (deliver|offer|respond|refund|pay|repair|replace))\b"), 3),
    ],
    "payment": [
        (_rx(r"\b(paid|payment|pay(ing)?|instalments?|installments?|deposit(ed)?|booking amount|advance|loan|sanctioned|disbursed|premium|remitted|transferred)\b"), 2),
        (MONEY, 2),
    ],
    "transaction": [
        (_rx(r"\b(booked|purchased|bought|allot(ted|ment)|agreement|executed|sale deed|buyer'?s agreement|hired|availed|subscribed|insured|admitted|ordered|placed an order|took (a )?(loan|policy))\b"), 3),
        (_rx(r"\b(project|flat|apartment|unit|plot|vehicle|policy|product|service)\b.{0,60}\b(booked|purchased|allotted|issued)"), 2),
        (_rx(r"\b(possession|delivery|completion)\b.{0,50}\b(within|period of|by)\b"), 2),
    ],
}

# Text after these markers is the court's operative order = the outcome.
OUTCOME_MARKERS = re.compile(
    r"disposed of with the following directions"
    r"|it is (hereby )?(ordered|directed)"
    r"|\bwe (hereby )?(allow|dismiss|direct)\b"
    r"|complaint is (partly |partially )?(allowed|dismissed)"
    r"|the (op|opposite part(y|ies)|respondents?) (shall|is directed to) (refund|pay)",
    re.IGNORECASE,
)

ABBREVIATIONS = {
    "vs", "v", "pvt", "ltd", "no", "nos", "rs", "dr", "mr", "mrs", "ms", "hon", "sec", "ors",
    "anr", "co", "inc", "smt", "shri", "m/s", "etc", "viz", "ex", "hon'ble", "cc", "ref", "art",
    "i.e", "e.g", "i.e.,", "approx", "dt",
}

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "real_estate": ["flat", "apartment", "builder", "developer", "possession", "allotment",
                    "project", "plot", "rera", "buyer's agreement", "carpet area", "unit"],
    "banking_finance": ["bank", "loan", "emi", "credit card", "atm", "account", "interest rate", "nbfc", "cheque"],
    "insurance": ["insurance", "insurer", "policy", "premium", "insured", "repudiat", "mediclaim", "sum assured"],
    "medical_negligence": ["hospital", "doctor", "surgery", "treatment", "patient", "negligence", "diagnosis", "nursing home"],
    "ecommerce_retail": ["online", "e-commerce", "seller", "order", "delivered", "courier", "website", "platform"],
    "product_defect": ["defect", "defective", "manufactur", "warranty", "vehicle", "car ", "mobile", "appliance", "service centre"],
    "telecom_utility": ["telecom", "broadband", "mobile connection", "electricity", "billing", "meter", "connection"],
    "education": ["university", "college", "admission", "tuition", "coaching", "course fee"],
    "travel_hospitality": ["airline", "flight", "hotel", "booking cancelled", "tour", "railway", "ticket"],
}

# CPA-2019 concept hooks: hints for the Statute Retrieval Agent. CiteVerify,
# not this agent, remains responsible for confirming any actual citation.
CPA_HOOKS = {
    "deficiency_in_service": {"cpa_2019_ref": "s.2(11)", "cues": r"deficien|delay|failed to|negligen|not (complete|deliver|offer)"},
    "defect_in_goods": {"cpa_2019_ref": "s.2(10)", "cues": r"defect|not as (promised|described)|manufactur|malfunction"},
    "unfair_trade_practice": {"cpa_2019_ref": "s.2(47)", "cues": r"misrepresent|false (claim|advert)|overcharg|unfair|refus(ed|al) to (refund|replace)"},
    "limitation_issue": {"cpa_2019_ref": "s.69", "cues": r"limitation|time[- ]barred|barred by"},
    "pecuniary_jurisdiction_issue": {"cpa_2019_ref": "s.34/47/58", "cues": r"pecuniary jurisdiction"},
}


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------
@dataclass
class Fact:
    id: str
    paragraph: int | None
    text: str
    span: tuple[int, int]
    label: str
    confidence: float
    secondary_labels: list = field(default_factory=list)
    dates: list = field(default_factory=list)          # ISO strings
    amounts_inr: list = field(default_factory=list)    # ints (rupees)
    durations: list = field(default_factory=list)      # {"value": float, "unit": str}
    percentages: list = field(default_factory=list)
    grounded: bool = True
    labelled_by: str = "rules"


@dataclass
class TimelineEvent:
    date: str
    fact_id: str
    event: str
    label: str


@dataclass
class DerivedFact:
    name: str
    value: str
    explanation: str
    inputs: list
    derived: bool = True


@dataclass
class FactSheet:
    case_number: str | None = None
    forum: str | None = None
    reference_date: str | None = None
    complainants: list = field(default_factory=list)
    opposite_parties: list = field(default_factory=list)
    dispute_category: str | None = None
    category_confidence: float = 0.0
    facts: list = field(default_factory=list)            # transaction/payment/deficiency
    claims: list = field(default_factory=list)
    defences: list = field(default_factory=list)
    procedural_points: list = field(default_factory=list)
    legal_reasoning_count: int = 0
    timeline: list = field(default_factory=list)
    total_amount_paid_inr: int | None = None
    derived_facts: list = field(default_factory=list)
    legal_hooks: list = field(default_factory=list)
    retrieval_query: str = ""
    outcome_removed: bool = False
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        L = [f"Case: {self.case_number}  |  Forum: {self.forum}  |  Reference date: {self.reference_date}"]
        L.append(f"Complainant(s): {', '.join(self.complainants) or '-'}")
        L.append(f"Opposite party(ies): {', '.join(self.opposite_parties) or '-'}")
        L.append(f"Dispute category: {self.dispute_category} ({self.category_confidence:.0%})")
        L.append(f"Outcome removed before extraction: {self.outcome_removed}")
        L.append(f"\nFACTS ({len(self.facts)})")
        for f in self.facts:
            L.append(f"  [{f.id} ¶{f.paragraph} {f.label} {f.confidence:.2f}] {f.text[:150]}")
        L.append(f"\nCOMPLAINANT'S CLAIMS ({len(self.claims)})")
        for f in self.claims:
            L.append(f"  [{f.id} ¶{f.paragraph}] {f.text[:150]}")
        L.append(f"\nOPPOSITE PARTY'S DEFENCES ({len(self.defences)})")
        for f in self.defences:
            L.append(f"  [{f.id} ¶{f.paragraph}] {f.text[:150]}")
        L.append(f"\nPROCEDURAL POINTS ({len(self.procedural_points)})")
        for f in self.procedural_points:
            L.append(f"  [{f.id} ¶{f.paragraph}] {f.text[:150]}")
        L.append("\nTIMELINE")
        for e in self.timeline:
            L.append(f"  {e.date}  ({e.fact_id}, {e.label})")
        L.append("\nDERIVED (computed, not stated in text)")
        for d in self.derived_facts:
            L.append(f"  {d.name}: {d.value} -- {d.explanation}")
        if self.total_amount_paid_inr:
            L.append(f"\nLargest amount paid: Rs.{self.total_amount_paid_inr:,}")
        L.append(f"\nLegal hooks for Statute Retrieval Agent: {[h['concept'] for h in self.legal_hooks]}")
        L.append(f"Retrieval query: {self.retrieval_query}")
        for w in self.warnings:
            L.append(f"WARNING: {w}")
        return "\n".join(L)


# --------------------------------------------------------------------------
# Helper functions (pure, individually testable)
# --------------------------------------------------------------------------
def parse_dates(text: str) -> list[str]:
    found: list[tuple[int, str]] = []
    for m in DATE_NUMERIC.finditer(text):
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            found.append((m.start(), date(y, mo, d).isoformat()))
        except ValueError:
            pass
    for m in DATE_WORDY.finditer(text):
        mo = MONTHS.get(m.group(2).lower())
        try:
            found.append((m.start(), date(int(m.group(3)), mo, int(m.group(1))).isoformat()))
        except (ValueError, TypeError):
            pass
    found.sort()
    return list(dict.fromkeys(iso for _, iso in found))


def parse_amounts(text: str) -> list[int]:
    out = []
    for m in MONEY.finditer(text):
        raw = m.group(1).replace(",", "").rstrip(".")
        if not raw:
            continue
        try:
            val = float(raw)
        except ValueError:
            continue
        unit = (m.group(2) or "").lower()
        out.append(int(round(val * UNIT_MULT.get(unit, 1))))
    return out


def parse_durations(text: str) -> list[dict]:
    out = []
    for m in DURATION.finditer(text):
        raw = m.group(1).lower()
        if raw in NUMBER_WORDS:
            val = float(NUMBER_WORDS[raw])
        elif " " in raw:  # "3 1/2"
            whole, frac = raw.split()
            n, d = frac.split("/")
            val = int(whole) + int(n) / int(d)
        else:
            val = float(raw)
        unit = m.group(2).lower().rstrip("s")
        out.append({"value": val, "unit": unit})
    return out


def add_months(d: date, months: int) -> date:
    y, m = divmod(d.month - 1 + months, 12)
    y += d.year
    m += 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def split_sentences(text: str, start: int = 0) -> list[dict]:
    """
    Sentence-split `text[start:]`, keeping absolute offsets into `text`.
    Tracks the numbered paragraph ("5.  As far as ...") each sentence sits in.
    Does not split after legal abbreviations (Vs., Pvt., Ltd., Rs., No. ...).
    """
    sentences = []
    para = 1  # an un-numbered opening paragraph is paragraph 1
    boundary = re.compile(r"(?<=[.;])\s+(?=[A-Z0-9(\"'‘“])")
    pos = 0
    for line in text[start:].split("\n"):
        line_start = start + pos
        pos += len(line) + 1
        stripped = line.strip()
        if len(stripped) < 25:
            continue
        pm = re.match(r"\s*(\d{1,3})\.\s+", line)
        if pm:
            para = int(pm.group(1))
        cursor = 0
        cuts = []
        for b in boundary.finditer(line):
            before = line[cursor:b.start()].split()
            last = before[-1].rstrip(".;").lower() if before else ""
            if last in ABBREVIATIONS or len(last) == 1 or re.fullmatch(r"\(?[ivx]+\)?", last):
                continue
            cuts.append(b.start())
        edges = [0] + [c + 1 for c in cuts] + [len(line)]
        for a, b in zip(edges, edges[1:]):
            seg = line[a:b]
            lead = len(seg) - len(seg.lstrip())
            seg_s = seg.strip()
            seg_s_clean = re.sub(r"^\d{1,3}\.\s+", "", seg_s) if a == 0 else seg_s
            if len(seg_s_clean) < 25:
                continue
            offset = line_start + a + lead + (len(seg_s) - len(seg_s_clean))
            sentences.append({
                "text": seg_s_clean,
                "start": offset,
                "end": offset + len(seg_s_clean),
                "paragraph": para,
            })
    return sentences


def find_body_start(text: str) -> int:
    """Offset just after the 'ORDER' / 'JUDGMENT' heading, else 0 (free-text case file)."""
    m = re.search(r"^\s*(ORDER|ORAL ORDER|JUDGMENT|FINAL ORDER)\s*$", text, re.IGNORECASE | re.MULTILINE)
    if not m:
        return 0
    nl = text.find("\n", m.end())
    # skip the bench line that often follows the heading, e.g. "JUSTICE V.K. JAIN, PRESIDING MEMBER (ORAL)"
    nxt = text[m.end():].lstrip("\n")
    first_line = nxt.split("\n", 1)[0]
    if re.search(r"(PRESIDING MEMBER|PRESIDENT|MEMBER)\b.*(\(ORAL\))?\s*$", first_line) and len(first_line) < 100:
        return text.find(first_line, m.end()) + len(first_line)
    return m.end()


def find_outcome_start(text: str, body_start: int) -> int | None:
    """Offset of the line where the operative order begins, or None."""
    m = OUTCOME_MARKERS.search(text, body_start)
    if not m:
        return None
    line_start = text.rfind("\n", 0, m.start()) + 1
    return line_start


def parse_parties(text: str) -> tuple[list[str], list[str]]:
    """Parse numbered party names around 'Complainant(s)' / 'Versus' / 'Opp.Party(s)'."""
    header_end = find_body_start(text) or len(text)
    header = text[:header_end]
    vs = re.search(r"^\s*(Versus|Vs\.?)\s*$", header, re.IGNORECASE | re.MULTILINE)
    if not vs:
        return [], []
    before, after = header[:vs.start()], header[vs.end():]
    after = re.split(r"\bBEFORE\b", after, maxsplit=1)[0]

    def names(block: str) -> list[str]:
        out = []
        for line in block.split("\n"):
            m = re.match(r"^\s*\d{1,2}\.\s+(.+?)\s*$", line)
            if m and not re.search(r"consumer case|appeal|complaint no", m.group(1), re.IGNORECASE):
                out.append(re.sub(r"\s+", " ", m.group(1)).strip())
        return out

    return names(before), names(after)


def detect_category(text: str) -> tuple[str | None, float]:
    low = text.lower()
    scores = {c: sum(low.count(k) for k in kws) for c, kws in CATEGORY_KEYWORDS.items()}
    total = sum(scores.values())
    if not total:
        return None, 0.0
    best = max(scores, key=scores.get)
    return best, round(scores[best] / total, 2)


def rule_classify(sentence: str) -> tuple[str, float, list[str]]:
    scores = {}
    for label, rules in RULES.items():
        s = sum(w for rx, w in rules if rx.search(sentence))
        if s:
            scores[label] = s
    if not scores:
        return "other", 0.3, []
    # A sentence naming a court/authority is reasoning even if it also mentions a delay.
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], ALL_LABELS.index(kv[0])))
    label, top = ranked[0]
    total = sum(scores.values())
    conf = round(min(0.95, 0.4 + 0.6 * (top / total) * min(1.0, top / 4)), 2)
    return label, conf, [l for l, _ in ranked[1:3]]


# --------------------------------------------------------------------------
# The agent
# --------------------------------------------------------------------------
LLM_SYSTEM = (
    "You are a legal text classifier for Indian consumer-dispute case files. "
    "Classify each numbered sentence into exactly one label from: "
    + ", ".join(ALL_LABELS) + ". "
    "transaction = the purchase/booking/agreement/service arrangement; "
    "payment = money paid/loans/instalments; deficiency = delay, defect, failure, non-performance; "
    "claim = what the complainant asks for; defence = opposite party's contention; "
    "procedural = limitation/jurisdiction/notice; "
    "legal_reasoning = the tribunal's discussion of law or precedent; other = anything else. "
    'Respond ONLY with JSON: [{"id": <int>, "label": "<label>", "confidence": <0-1>}]. '
    "Do not add, alter or invent text."
)


class FactExtractionAgent(BaseAgent):
    name = "fact_extraction_agent"

    def __init__(self, llm=None, strip_outcome: bool = True):
        """
        llm: optional backend (see agents/llm.py). None = fully offline rules.
        strip_outcome: drop the operative order before extraction (leakage control).
        """
        self.llm = llm
        self.strip_outcome = strip_outcome

    # ---- pipeline interface -------------------------------------------------
    def run(self, state: dict) -> dict:
        sheet = self.extract(state["case_text"], state.get("query", ""))
        state["fact_sheet"] = sheet
        state["retrieval_query"] = sheet.retrieval_query
        state["legal_hooks"] = sheet.legal_hooks
        return state

    # ---- main entry ---------------------------------------------------------
    def extract(self, case_text: str, query: str = "") -> FactSheet:
        sheet = FactSheet()
        if not case_text or len(case_text.strip()) < 50:
            sheet.warnings.append("Case text is empty or too short to extract facts.")
            return sheet

        # metadata (reuse Stage-2 detectors so both stages agree)
        from structuring.segment import detect_forum, CASE_NUMBER_PATTERN
        sheet.forum = detect_forum(case_text)
        cn = CASE_NUMBER_PATTERN.search(case_text)
        sheet.case_number = cn.group(1) if cn else None
        dm = re.search(r"Dated?\s*:?\s*(.+)", case_text)
        if dm:
            found = parse_dates(dm.group(1))
            sheet.reference_date = found[0] if found else None
        sheet.complainants, sheet.opposite_parties = parse_parties(case_text)

        # body + outcome-leakage control
        body_start = find_body_start(case_text)
        end = len(case_text)
        if self.strip_outcome:
            cut = find_outcome_start(case_text, body_start)
            if cut is not None:
                end = cut
                sheet.outcome_removed = True
        working = case_text[:end]

        sents = split_sentences(working, body_start)
        if not sents:
            sheet.warnings.append("No sentences found after isolating the body.")
            return sheet

        labels = self._classify(sents)

        counter = 0
        for s, (label, conf, secondary, by) in zip(sents, labels):
            if label == "legal_reasoning":
                sheet.legal_reasoning_count += 1
                continue
            if label == "other":
                continue
            counter += 1
            f = Fact(
                id=f"F{counter}", paragraph=s["paragraph"], text=s["text"],
                span=(s["start"], s["end"]), label=label, confidence=conf,
                secondary_labels=secondary,
                dates=parse_dates(s["text"]), amounts_inr=parse_amounts(s["text"]),
                durations=parse_durations(s["text"]), percentages=[float(p) for p in PERCENT.findall(s["text"])],
                labelled_by=by,
            )
            f.grounded = case_text[f.span[0]:f.span[1]] == f.text   # transparency check
            if not f.grounded:
                sheet.warnings.append(f"{f.id} failed grounding check and was dropped.")
                continue
            if label in FACT_LABELS:
                sheet.facts.append(f)
            elif label == "claim":
                sheet.claims.append(f)
            elif label == "defence":
                sheet.defences.append(f)
            elif label == "procedural":
                sheet.procedural_points.append(f)

        self._build_timeline(sheet)
        self._largest_payment(sheet)
        self._derive(sheet)
        sheet.dispute_category, sheet.category_confidence = detect_category(
            " ".join(f.text for f in sheet.facts + sheet.claims) or working)
        self._hooks_and_query(sheet, query)

        if not sheet.facts:
            sheet.warnings.append("No factual sentences identified - is this a narrative case file?")
        if not sheet.complainants:
            sheet.warnings.append("Party names not found (free-text input?); parties left empty.")
        return sheet

    # ---- classification -----------------------------------------------------
    def _classify(self, sents: list[dict]) -> list[tuple]:
        base = []
        for s in sents:
            label, conf, sec = rule_classify(s["text"])
            base.append([label, conf, sec, "rules"])
        if self.llm is None:
            return [tuple(b) for b in base]
        try:
            payload = "\n".join(f"{i}: {s['text']}" for i, s in enumerate(sents))
            reply = self.llm.complete_json(LLM_SYSTEM, payload)
            for item in reply:
                i, lab = int(item["id"]), item["label"]
                if 0 <= i < len(base) and lab in ALL_LABELS:
                    base[i] = [lab, round(float(item.get("confidence", 0.7)), 2), [base[i][0]], "llm"]
        except Exception as e:  # noqa: BLE001 - any failure => rule fallback
            self._llm_error = str(e)
        return [tuple(b) for b in base]

    # ---- structuring --------------------------------------------------------
    @staticmethod
    def _build_timeline(sheet: FactSheet):
        events = []
        for f in sheet.facts + sheet.claims + sheet.defences + sheet.procedural_points:
            for d in f.dates:
                events.append(TimelineEvent(date=d, fact_id=f.id, event=f.text, label=f.label))
        # de-dup identical (date, fact) and sort chronologically
        seen, uniq = set(), []
        for e in sorted(events, key=lambda e: (e.date, e.fact_id)):
            if (e.date, e.fact_id) not in seen:
                seen.add((e.date, e.fact_id))
                uniq.append(e)
        sheet.timeline = uniq

    @staticmethod
    def _largest_payment(sheet: FactSheet):
        amounts = [a for f in sheet.facts if f.label in ("payment", "deficiency", "transaction") for a in f.amounts_inr]
        # "paid Rs.X" style statements only: avoids picking up sale price or jurisdiction limits
        paid = [a for f in sheet.facts + sheet.claims
                if re.search(r"\b(paid|payment|deposited)\b", f.text, re.IGNORECASE) for a in f.amounts_inr]
        sheet.total_amount_paid_inr = max(paid) if paid else (max(amounts) if amounts else None)

    @staticmethod
    def _derive(sheet: FactSheet):
        """Computed facts. Always labelled derived=True with the inputs used."""
        agreement = None
        for f in sheet.facts:
            if re.search(r"\bagreement\b.*\bexecuted\b|\bexecuted\b.*\bagreement\b", f.text, re.IGNORECASE) and f.dates:
                agreement = (f, f.dates[-1])
                break
        promise = None
        for f in sheet.facts:
            if re.search(r"\b(possession|deliver\w*|completion|handover)\b", f.text, re.IGNORECASE):
                for du in f.durations:
                    if du["unit"] in ("month", "year"):
                        promise = (f, du)
                        break
            if promise:
                break
        if not (agreement and promise):
            return
        af, adate = agreement
        pf, du = promise
        months = int(round(du["value"] * (12 if du["unit"] == "year" else 1)))
        deadline = add_months(date.fromisoformat(adate), months)
        sheet.derived_facts.append(DerivedFact(
            name="promised_delivery_deadline", value=deadline.isoformat(),
            explanation=f"agreement date {adate} + {months} months (as promised in {pf.id})",
            inputs=[af.id, pf.id],
        ))
        if sheet.reference_date:
            ref = date.fromisoformat(sheet.reference_date)
            days = (ref - deadline).days
            if days > 0:
                sheet.derived_facts.append(DerivedFact(
                    name="delay_days_at_reference_date", value=str(days),
                    explanation=f"{deadline.isoformat()} to {ref.isoformat()} (~{days / 365.25:.1f} years)",
                    inputs=[af.id, pf.id],
                ))
            else:
                sheet.derived_facts.append(DerivedFact(
                    name="delay_days_at_reference_date", value="0",
                    explanation="deadline had not passed at the reference date", inputs=[af.id, pf.id]))

    @staticmethod
    def _hooks_and_query(sheet: FactSheet, query: str):
        corpus = " ".join(f.text for f in sheet.facts + sheet.claims + sheet.defences + sheet.procedural_points)
        corpus += " " + query
        hooks = []
        for concept, meta in CPA_HOOKS.items():
            hits = re.findall(meta["cues"], corpus, re.IGNORECASE)
            if hits:
                hooks.append({"concept": concept, "cpa_2019_ref": meta["cpa_2019_ref"], "cue_hits": len(hits)})
        sheet.legal_hooks = sorted(hooks, key=lambda h: -h["cue_hits"])

        parts = []
        if sheet.dispute_category:
            parts.append(sheet.dispute_category.replace("_", " "))
        parts += [h["concept"].replace("_", " ") for h in sheet.legal_hooks[:3]]
        if any(re.search(r"\brefund\b", f.text, re.IGNORECASE) for f in sheet.claims + sheet.facts):
            parts.append("refund")
        if any(re.search(r"\bcompensation|interest\b", f.text, re.IGNORECASE) for f in sheet.claims):
            parts.append("compensation interest")
        if query.strip():
            parts.append(query.strip())
        sheet.retrieval_query = " ".join(dict.fromkeys(" ".join(parts).split()))
