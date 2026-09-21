"""
Agent 3 of the Multi-Agent Legal Reasoning Pipeline: PRECEDENT RETRIEVAL AGENT.

Objective (project synopsis): retrieve legal evidence for the case under
consideration, so the Argument Analysis and Prediction agents reason over
*verified, grounded* prior decisions instead of recalled or hallucinated
case law.

Position in the pipeline
------------------------
    Fact Extraction -> Statute Retrieval -> [PRECEDENT RETRIEVAL] ->
    Argument Analysis -> Prediction -> Verification -> Explanation

It consumes the `fact_sheet` written by the Fact Extraction Agent (and, when
present, the `statutes` written by the Statute Retrieval Agent) and writes a
`precedent_bundle` into the shared state.

What it does
------------
1. Builds a precedent-specific retrieval query from the fact sheet: dispute
   category + CPA-2019 concept hooks + fact-pattern signals + relief sought.
   This is deliberately *not* the raw user query - a precedent is relevant
   because of its fact pattern and legal issue, not its wording.
2. Retrieves candidates from the structured legal DB through the existing
   Hybrid Retriever (BM25 + dense fusion), with a BM25-only fallback for
   corpora too small to fit a dense index.
3. Excludes the case under consideration from its own results
   (`_self_exclusion`). If the input is a decided judgment already sitting in
   the corpus, retrieving it would hand the Prediction Agent the answer. This
   is the same outcome-leakage control the Fact Extraction Agent applies when
   it strips the operative order.
4. Re-ranks candidates on six transparent, individually reported signals:
   retrieval score, dispute-category match, statutory/hook overlap,
   fact-pattern signal overlap, forum authority, and recency. Every weight is
   in `SCORING_WEIGHTS` and every contribution is reported in
   `PrecedentMatch.why_relevant` - no opaque score.
5. For each match, extracts the legal evidence the downstream agents need:
      - holdings (candidate ratio decidendi sentences), grounded by offsets
      - the operative outcome (allowed / partly allowed / dismissed / ...)
      - the relief granted (amounts, interest rate, relief types)
      - alignment: does this authority help the complainant or the OP
      - distinguishing factors: where the precedent's facts diverge
6. Verification-first (Objective 4): every precedent surfaced, and every
   authority cited *inside* it, is run through CiteVerify before it can be
   relied on. Unverifiable citations are surfaced as UNVERIFIED with a
   reason, never silently dropped and never silently passed.
7. Aggregates an outcome distribution and an authority-weighted complainant
   success rate - the prior the Prediction Agent conditions on.

Rules vs. LLM
-------------
Default is offline and rule-based (same trade-off documented in the README
for NER / IRAC / Fact Extraction). Pass `llm=AnthropicBackend()` to have an
LLM re-score the shortlist for legal relevance. The LLM may only re-order and
annotate candidates that retrieval already found in the corpus - it can never
introduce a case, so it cannot hallucinate an authority. Any failure falls
back to the rule scores.
"""
from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass, field, asdict
from datetime import date

from agents.base import BaseAgent
from agents.fact_extraction import (
    FactSheet,
    PERCENT,
    find_body_start,
    find_outcome_start,
    parse_amounts,
    parse_dates,
    split_sentences,
)
from structuring.segment import detect_forum, extract_statutes
from verification.citeverify import verify_precedent

# --------------------------------------------------------------------------
# Forum authority. A National Commission decision binds a State Commission;
# a Supreme Court decision binds everyone. Precedential weight has to reflect
# that hierarchy, otherwise the Prediction Agent treats a District Forum order
# as equal authority to a Supreme Court ruling.
# --------------------------------------------------------------------------
FORUM_AUTHORITY = {
    "Supreme Court": 1.00,
    "High Court": 0.90,
    "NCDRC": 0.85,
    "State Commission": 0.60,
    "District Forum": 0.40,
}
DEFAULT_AUTHORITY = 0.50

SUPREME_COURT_PATTERN = re.compile(r"\bIN THE SUPREME COURT OF INDIA\b", re.IGNORECASE)
HIGH_COURT_PATTERN = re.compile(r"\bIN THE HIGH COURT OF\b", re.IGNORECASE)

# --------------------------------------------------------------------------
# Fact-pattern signals. Two consumer disputes are "alike" for precedent
# purposes when they share these issue-level features - not when they share
# vocabulary. Kept separate from the Fact Extraction Agent's CPA hooks:
# hooks answer "which provision", signals answer "which fact pattern".
# --------------------------------------------------------------------------
SIGNAL_PATTERNS: dict[str, re.Pattern] = {
    "delayed_possession": re.compile(
        r"\b(possession|handover|hand over)\b.{0,60}\b(delay|not (delivered|offered|handed)|never)\b"
        r"|\bdelay(ed)? (in )?(possession|delivery|handover)\b"
        # "did not deliver or even offer the possession of the allotted flat"
        r"|\b(did not|failed to|never|neither) (deliver|offer|hand over)\b.{0,60}\bpossession\b"
        # "possession was to be delivered within 36 months" + the unit never arrived
        r"|\bpossession\b.{0,60}\b(was|is) (still )?(not|yet to be)\b",
        re.IGNORECASE),
    "non_delivery": re.compile(
        r"\b(did not|failed to|never) (deliver|dispatch|supply|hand over)\b|\bnon[- ]delivery\b", re.IGNORECASE),
    "refund_sought": re.compile(r"\brefund\b", re.IGNORECASE),
    "compensation_interest": re.compile(
        r"\b(compensation|interest @|interest at|per annum|damages)\b", re.IGNORECASE),
    "defective_goods": re.compile(
        r"\b(defect(ive|s)?|manufacturing defect|malfunction|not as (promised|described))\b", re.IGNORECASE),
    "service_deficiency": re.compile(
        r"\b(deficien\w+ in service|deficien\w+|negligen\w+)\b", re.IGNORECASE),
    "claim_repudiation": re.compile(r"\brepudiat\w+|claim (was )?(rejected|denied)\b", re.IGNORECASE),
    "unfair_trade_practice": re.compile(
        r"\bunfair trade practice|misrepresent\w+|false (advertis|claim)|overcharg\w+\b", re.IGNORECASE),
    "limitation_defence": re.compile(r"\blimitation\b|\btime[- ]barred\b|\bbarred by limitation\b", re.IGNORECASE),
    "jurisdiction_defence": re.compile(
        r"\b(pecuniary|territorial) jurisdiction\b|\bmaintainab\w+\b", re.IGNORECASE),
    "force_majeure_defence": re.compile(
        r"\bforce majeure\b|\bcircumstances beyond (its|their|our) control\b", re.IGNORECASE),
    "builder_buyer_agreement": re.compile(
        r"\b(buyer'?s? agreement|builder[- ]buyer|allotment letter|flat buyer)\b", re.IGNORECASE),
    "financing_bank_involved": re.compile(
        r"\b(loan|emi|disburs\w+|tripartite|subvention|home loan)\b", re.IGNORECASE),
    "medical_treatment": re.compile(r"\b(hospital|surgery|treatment|patient|diagnos\w+)\b", re.IGNORECASE),
    "insurance_policy": re.compile(r"\b(policy|premium|insured|sum assured|mediclaim)\b", re.IGNORECASE),
}

# --------------------------------------------------------------------------
# Holding (ratio decidendi) cues. A holding is where the tribunal states the
# legal proposition it is deciding on - distinct from narrative facts and
# from the operative directions.
# --------------------------------------------------------------------------
HOLDING_CUES: list[tuple[re.Pattern, float]] = [
    (re.compile(r"\b(amounts? to|constitutes?|tantamount to)\b.{0,60}\b(deficiency|unfair trade practice|negligence)\b",
                re.IGNORECASE), 3.0),
    (re.compile(r"\b(we are of the (considered )?(view|opinion)|in our (considered )?(view|opinion))\b",
                re.IGNORECASE), 3.0),
    (re.compile(r"\b(it is (well )?settled|settled law|held (that|by)|has been held)\b", re.IGNORECASE), 2.5),
    (re.compile(r"\b(cannot be compelled|cannot be made to wait|is entitled to|are entitled to)\b",
                re.IGNORECASE), 2.0),
    (re.compile(r"\bin terms of section\b|\bas defined (by|in) section\b|\bunder section \d", re.IGNORECASE), 1.5),
    (re.compile(r"\b(recurrent cause of action|continuing cause of action)\b", re.IGNORECASE), 2.0),
    (re.compile(r"\b(hon'?ble supreme court|apex court|larger bench)\b", re.IGNORECASE), 1.0),
]

# --------------------------------------------------------------------------
# Operative outcome classification. Ordered: the first pattern that matches
# in the operative region wins, so "partly allowed" is tested before "allowed".
# --------------------------------------------------------------------------
OUTCOME_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("partly_allowed", re.compile(
        r"\b(part(ly|ially) allowed|allowed in part|partly succeeds)\b", re.IGNORECASE)),
    ("dismissed", re.compile(
        r"\b(complaint|appeal|revision petition|petition) (is|stands|are) dismissed\b"
        r"|\bwe dismiss\b|\bdismissed with no order as to cost\b|\bhereby dismissed\b", re.IGNORECASE)),
    ("allowed", re.compile(
        r"\b(complaint|appeal|revision petition) (is|stands) (hereby )?allowed\b|\bwe allow\b", re.IGNORECASE)),
    ("disposed_with_directions", re.compile(
        r"\bdisposed of with the following directions?\b"
        r"|\b(shall|is directed to|are directed to) (refund|pay|replace|deliver)\b"
        r"|\bit is (hereby )?(ordered|directed)\b", re.IGNORECASE)),
]

RELIEF_PATTERNS: dict[str, re.Pattern] = {
    "refund": re.compile(r"\brefund\b", re.IGNORECASE),
    "interest": re.compile(r"\binterest\b", re.IGNORECASE),
    "compensation": re.compile(r"\bcompensation\b", re.IGNORECASE),
    "litigation_cost": re.compile(r"\b(cost of litigation|litigation (cost|expenses)|costs? of\b)", re.IGNORECASE),
    "replacement": re.compile(r"\breplace(ment)?\b", re.IGNORECASE),
    "possession": re.compile(r"\b(deliver|hand over) (the )?possession\b", re.IGNORECASE),
}

# Outcomes that granted the consumer something.
COMPLAINANT_FAVOURABLE = {"allowed", "partly_allowed", "disposed_with_directions"}

# --------------------------------------------------------------------------
# Re-ranking weights. Exposed as a module constant so the review panel (and
# the ablation table in the report) can see and vary them.
# --------------------------------------------------------------------------
SCORING_WEIGHTS = {
    "retrieval": 0.40,      # hybrid BM25 + dense score
    "category": 0.15,       # same dispute category
    "hooks": 0.15,          # shared CPA-2019 provisions / statutory basis
    "signals": 0.15,        # shared fact-pattern features
    "authority": 0.10,      # forum hierarchy
    "recency": 0.05,        # newer decisions reflect current law
}

RECENCY_HALF_LIFE_YEARS = 8.0


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------
@dataclass
class Holding:
    """A candidate ratio-decidendi sentence, grounded in the source judgment."""
    text: str
    paragraph: int | None
    span: tuple[int, int]
    cue_score: float
    grounded: bool = True


@dataclass
class Outcome:
    """The operative result of a precedent, read from its order region."""
    label: str = "unknown"                      # see OUTCOME_PATTERNS
    favours: str = "unknown"                    # complainant | opposite_party | unknown
    evidence: str = ""                          # verbatim sentence the label came from
    span: tuple[int, int] | None = None
    relief_types: list = field(default_factory=list)
    amounts_inr: list = field(default_factory=list)
    interest_rates: list = field(default_factory=list)


@dataclass
class CitedAuthority:
    """A case cited *inside* a retrieved precedent (citation-graph expansion)."""
    case_name: str
    year: str | None = None
    reporter_citation: str | None = None
    cited_in_judgment_id: int | None = None
    verified: bool = False
    verification_confidence: float = 0.0
    verification_reason: str = ""


@dataclass
class PrecedentMatch:
    judgment_id: int | None
    case_number: str | None
    forum: str | None
    date: str | None
    final_score: float = 0.0
    score_components: dict = field(default_factory=dict)
    retrieval_scores: dict = field(default_factory=dict)
    dispute_category: str | None = None
    shared_hooks: list = field(default_factory=list)
    shared_signals: list = field(default_factory=list)
    distinguishing_factors: list = field(default_factory=list)
    holdings: list = field(default_factory=list)          # list[Holding]
    outcome: Outcome = field(default_factory=Outcome)
    cited_authorities: list = field(default_factory=list)  # list[CitedAuthority]
    why_relevant: list = field(default_factory=list)
    verified: bool = False
    verification_confidence: float = 0.0
    verification_reason: str = ""
    admissible_as_authority: bool = False
    ranked_by: str = "rules"
    llm_rationale: str | None = None
    snippet: str = ""


@dataclass
class PrecedentBundle:
    precedent_query: str = ""
    corpus_size: int = 0
    candidates_considered: int = 0
    matches: list = field(default_factory=list)            # list[PrecedentMatch]
    excluded_judgment_ids: list = field(default_factory=list)
    exclusion_reasons: dict = field(default_factory=dict)
    outcome_distribution: dict = field(default_factory=dict)
    complainant_success_rate: float | None = None
    authority_weighted_success_rate: float | None = None
    verified_count: int = 0
    unverified_count: int = 0
    retrieval_mode: str = "hybrid"
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        L = [f"Precedent query: {self.precedent_query}",
             f"Corpus: {self.corpus_size} judgment(s)  |  candidates considered: "
             f"{self.candidates_considered}  |  retrieval: {self.retrieval_mode}"]
        if self.excluded_judgment_ids:
            reasons = ", ".join(f"{jid} ({self.exclusion_reasons.get(str(jid), '?')})"
                                for jid in self.excluded_judgment_ids)
            L.append(f"Excluded from results (leakage control): {reasons}")
        L.append(f"Verified as authority: {self.verified_count}  |  unverified: {self.unverified_count}")
        L.append(f"\nRANKED PRECEDENTS ({len(self.matches)})")
        for i, m in enumerate(self.matches, 1):
            flag = "VERIFIED" if m.verified else "UNVERIFIED"
            L.append(f"\n{i}. [{m.final_score:.3f}] {m.case_number} ({m.forum}, {m.date})  [{flag}]")
            L.append(f"   components: " + "  ".join(f"{k}={v:.2f}" for k, v in m.score_components.items()))
            L.append(f"   outcome: {m.outcome.label} -> favours {m.outcome.favours}")
            if m.outcome.relief_types:
                L.append(f"   relief: {', '.join(m.outcome.relief_types)}"
                         + (f"  amounts={m.outcome.amounts_inr}" if m.outcome.amounts_inr else "")
                         + (f"  interest={m.outcome.interest_rates}" if m.outcome.interest_rates else ""))
            if m.shared_signals:
                L.append(f"   shared fact pattern: {', '.join(m.shared_signals)}")
            if m.distinguishing_factors:
                L.append(f"   distinguishable on: {', '.join(m.distinguishing_factors)}")
            for h in m.holdings:
                L.append(f"   holding (¶{h.paragraph}): {h.text[:170]}")
            for c in m.cited_authorities:
                mark = "verified" if c.verified else "UNVERIFIED"
                L.append(f"   cites: {c.case_name} [{mark} {c.verification_confidence:.2f}]")
            if m.llm_rationale:
                L.append(f"   llm: {m.llm_rationale}")
        L.append("\nEVIDENCE FOR PREDICTION AGENT")
        L.append(f"  outcome distribution: {self.outcome_distribution}")
        L.append(f"  complainant success rate: {self.complainant_success_rate}")
        L.append(f"  authority-weighted success rate: {self.authority_weighted_success_rate}")
        for w in self.warnings:
            L.append(f"WARNING: {w}")
        return "\n".join(L)


# --------------------------------------------------------------------------
# Helper functions (pure, individually testable)
# --------------------------------------------------------------------------
def detect_forum_extended(text: str) -> str | None:
    """`structuring.detect_forum` + the two apex fora, which outrank every consumer forum."""
    if SUPREME_COURT_PATTERN.search(text):
        return "Supreme Court"
    if HIGH_COURT_PATTERN.search(text):
        return "High Court"
    return detect_forum(text)


def authority_weight(forum: str | None) -> float:
    return FORUM_AUTHORITY.get(forum or "", DEFAULT_AUTHORITY)


def case_signals(text: str) -> set[str]:
    """The fact-pattern features present in a case text."""
    return {name for name, rx in SIGNAL_PATTERNS.items() if rx.search(text)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def recency_score(judgment_date: str | None, reference: date | None = None) -> float:
    """1.0 for a decision handed down today, decaying with an 8-year half-life."""
    if not judgment_date:
        return 0.5  # unknown date: neutral, never penalised to zero
    iso = parse_dates(judgment_date)
    if not iso:
        return 0.5
    try:
        d = date.fromisoformat(iso[0])
    except ValueError:
        return 0.5
    ref = reference or date.today()
    years = max(0.0, (ref - d).days / 365.25)
    return round(0.5 ** (years / RECENCY_HALF_LIFE_YEARS), 4)


def extract_holdings(text: str, max_holdings: int = 3) -> list[Holding]:
    """
    Locate the sentences most likely to carry the ratio decidendi.

    Extractive and grounded: each holding is a verbatim sentence of the source
    judgment with its offsets, re-sliced and checked. Nothing is generated, so
    a holding cannot be a paraphrase the tribunal never wrote.
    """
    body_start = find_body_start(text)
    scored: list[Holding] = []
    for s in split_sentences(text, body_start):
        score = sum(w for rx, w in HOLDING_CUES if rx.search(s["text"]))
        if score <= 0:
            continue
        h = Holding(text=s["text"], paragraph=s["paragraph"],
                    span=(s["start"], s["end"]), cue_score=score)
        h.grounded = text[h.span[0]:h.span[1]] == h.text
        if h.grounded:
            scored.append(h)
    scored.sort(key=lambda h: (-h.cue_score, h.span[0]))
    return scored[:max_holdings]


def extract_outcome(text: str) -> Outcome:
    """
    Read the operative result from the order region of a judgment.

    Uses the same boundary detector the Fact Extraction Agent uses for leakage
    control, so "what the agent hides from Prediction on the query case" and
    "what the agent reads off a precedent" are by construction the same region.
    """
    body_start = find_body_start(text)
    cut = find_outcome_start(text, body_start)
    region_start = cut if cut is not None else body_start
    region = text[region_start:]
    if not region.strip():
        return Outcome()

    out = Outcome()
    for label, rx in OUTCOME_PATTERNS:
        m = rx.search(region)
        if not m:
            continue
        line_start = region.rfind("\n", 0, m.start()) + 1
        line_end = region.find("\n", m.end())
        line_end = len(region) if line_end == -1 else line_end
        out.label = label
        out.evidence = re.sub(r"\s+", " ", region[line_start:line_end]).strip()[:300]
        out.span = (region_start + line_start, region_start + line_end)
        break

    out.favours = ("complainant" if out.label in COMPLAINANT_FAVOURABLE
                   else "opposite_party" if out.label == "dismissed" else "unknown")
    out.relief_types = [name for name, rx in RELIEF_PATTERNS.items() if rx.search(region)]
    out.amounts_inr = sorted(set(parse_amounts(region)), reverse=True)[:5]
    out.interest_rates = sorted({float(p) for p in PERCENT.findall(region)})
    if out.label == "dismissed":
        # nothing was granted: relief words in a dismissal are the rejected prayer
        out.relief_types, out.amounts_inr, out.interest_rates = [], [], []
    return out


def build_precedent_query(fact_sheet: FactSheet | None, user_query: str = "",
                          statutes: list[dict] | None = None) -> str:
    """
    A precedent is relevant for its issue and fact pattern, so the query is
    built from the structured fact sheet rather than the user's wording.
    Order: category -> statutory hooks -> fact-pattern signals -> relief.
    """
    parts: list[str] = []
    corpus = ""
    if fact_sheet is not None:
        if fact_sheet.dispute_category:
            parts.append(fact_sheet.dispute_category.replace("_", " "))
        parts += [h["concept"].replace("_", " ") for h in fact_sheet.legal_hooks[:3]]
        corpus = " ".join(f.text for f in (fact_sheet.facts + fact_sheet.claims + fact_sheet.defences))
    corpus = f"{corpus} {user_query}".strip()
    parts += [s.replace("_", " ") for s in sorted(case_signals(corpus))]
    for st in (statutes or []):
        section, act = st.get("section"), st.get("act", "")
        if section:
            parts.append(f"section {section}")
        if "consumer protection" in (act or "").lower():
            parts.append("consumer protection act")
    if user_query.strip():
        parts.append(user_query.strip())
    # de-duplicate tokens while preserving order
    return " ".join(dict.fromkeys(" ".join(parts).split()))


# --------------------------------------------------------------------------
# The agent
# --------------------------------------------------------------------------
LLM_SYSTEM = (
    "You are a legal research assistant for Indian consumer-dispute adjudication "
    "(Consumer Protection Act, 2019). You are given a summary of a pending case and a "
    "numbered shortlist of candidate precedents that a retrieval system already found in "
    "an authenticated corpus. Score how strongly each candidate governs the pending case, "
    "based on issue and fact-pattern similarity and on the forum's authority. "
    'Respond ONLY with JSON: [{"id": <int>, "relevance": <0-1>, "reason": "<one short sentence>"}]. '
    "Score only the candidates given. Never add a case, a citation, a section or a holding "
    "that is not in the shortlist."
)


class PrecedentRetrievalAgent(BaseAgent):
    """
    Agent 3: retrieves and ranks prior decisions that govern the pending case.

    Usage inside the pipeline (LangGraph-compatible, same contract as Agent 1):
        state = FactExtractionAgent().run(state)
        state = PrecedentRetrievalAgent(conn=conn).run(state)
        # -> state["precedent_bundle"], state["precedents"]
    """
    name = "precedent_retrieval_agent"

    def __init__(self, conn=None, db_path: str = "data_pipeline/legal.db",
                 top_k: int = 5, candidate_pool: int = 20,
                 llm=None, max_holdings: int = 3,
                 verify_threshold: float = 0.55,
                 exclude_self: bool = True,
                 self_similarity_threshold: float = 0.85,
                 min_score: float = 0.0):
        """
        conn:  open sqlite3 connection to the structured legal DB. If None, one
               is opened lazily from `db_path` at run time.
        exclude_self: drop the case under consideration from its own results.
               Leave this True for evaluation - a decided judgment sitting in
               the corpus would otherwise return itself and leak its outcome.
        min_score: floor on the final score; below it a candidate is dropped as
               not governing rather than padded into the result list.
        """
        self.conn = conn
        self.db_path = db_path
        self.top_k = top_k
        self.candidate_pool = candidate_pool
        self.llm = llm
        self.max_holdings = max_holdings
        self.verify_threshold = verify_threshold
        self.exclude_self = exclude_self
        self.self_similarity_threshold = self_similarity_threshold
        self.min_score = min_score
        self.llm_error: str | None = None

    # ---- pipeline interface -------------------------------------------------
    def run(self, state: dict) -> dict:
        bundle = self.retrieve(
            fact_sheet=state.get("fact_sheet"),
            case_text=state.get("case_text", ""),
            user_query=state.get("query", ""),
            statutes=state.get("statutes"),
            self_judgment_id=state.get("judgment_id"),
        )
        state["precedent_bundle"] = bundle
        state["precedents"] = bundle.matches
        state["precedent_query"] = bundle.precedent_query
        return state

    # ---- main entry ---------------------------------------------------------
    def retrieve(self, fact_sheet: FactSheet | None = None, case_text: str = "",
                 user_query: str = "", statutes: list[dict] | None = None,
                 self_judgment_id: int | None = None) -> PrecedentBundle:
        conn = self._get_conn()
        bundle = PrecedentBundle()
        bundle.precedent_query = build_precedent_query(fact_sheet, user_query, statutes)

        from structuring.db import fetch_all_judgments
        documents = fetch_all_judgments(conn)
        bundle.corpus_size = len(documents)
        if not documents:
            bundle.warnings.append("Legal DB is empty - ingest judgments before precedent retrieval.")
            return bundle
        if not bundle.precedent_query.strip():
            bundle.warnings.append("Empty precedent query - fact sheet had no category, hooks or signals.")
            return bundle

        # --- leakage control: the case under consideration is not its own precedent
        pool, excluded = self._self_exclusion(documents, case_text, fact_sheet, self_judgment_id)
        bundle.excluded_judgment_ids = [d["judgment_id"] for d, _ in excluded]
        bundle.exclusion_reasons = {str(d["judgment_id"]): r for d, r in excluded}
        if not pool:
            bundle.warnings.append(
                "Every judgment in the corpus was excluded as the case under consideration; "
                "no independent precedent available.")
            return bundle

        # --- stage 1: candidate retrieval
        ranked, mode = self._retrieve_candidates(pool, bundle.precedent_query)
        bundle.retrieval_mode = mode
        bundle.candidates_considered = len(ranked)
        if mode == "bm25_only":
            bundle.warnings.append(
                "Corpus too small for a dense index - ranked on BM25 alone (scores less reliable).")

        # --- stage 2: legal re-ranking
        query_signals, query_hooks, query_category = self._query_profile(fact_sheet, user_query, case_text)
        matches = [self._build_match(doc, scores, query_signals, query_hooks, query_category, fact_sheet)
                   for doc, scores in ranked]

        # --- stage 3: optional LLM re-scoring of the shortlist (never adds a case)
        shortlist = sorted(matches, key=lambda m: -m.final_score)[:self.candidate_pool]
        if self.llm is not None:
            self._llm_rescore(shortlist, fact_sheet, bundle)

        shortlist = [m for m in shortlist if m.final_score >= self.min_score]
        shortlist.sort(key=lambda m: -m.final_score)
        selected = shortlist[:self.top_k]

        # --- stage 4: verification-first (Objective 4)
        for m in selected:
            self._verify(m, conn)
            self._attach_cited_authorities(m, conn)

        bundle.matches = selected
        bundle.verified_count = sum(1 for m in selected if m.verified)
        bundle.unverified_count = len(selected) - bundle.verified_count
        self._aggregate_evidence(bundle)

        if not selected:
            bundle.warnings.append(
                f"No candidate scored at or above min_score={self.min_score}; "
                "the corpus may not contain a governing precedent for this fact pattern.")
        if bundle.corpus_size < 5:
            bundle.warnings.append(
                f"Only {bundle.corpus_size} judgment(s) in the corpus - the outcome distribution "
                "below is indicative, not a statistical prior.")
        return bundle

    # ---- internals ----------------------------------------------------------
    def _get_conn(self):
        if self.conn is None:
            from structuring.db import get_conn
            self.conn = get_conn(self.db_path)
        return self.conn

    def _self_exclusion(self, documents, case_text, fact_sheet, self_judgment_id):
        """
        Drop the case under consideration from the candidate pool.

        Three independent tests, because the caller may identify the case in
        three ways: by judgment_id, by case number, or not at all (raw text
        pasted in that happens to already be in the corpus).
        """
        if not self.exclude_self:
            return list(documents), []

        pool, excluded = [], []
        head = re.sub(r"\s+", " ", (case_text or "")[:1200]).strip().lower()
        case_number = (fact_sheet.case_number if fact_sheet else None) or ""
        for d in documents:
            if self_judgment_id is not None and d["judgment_id"] == self_judgment_id:
                excluded.append((d, "same judgment_id as the case under consideration"))
                continue
            if case_number and d.get("case_number") and \
                    case_number.strip().lower() == d["case_number"].strip().lower():
                excluded.append((d, "same case number as the case under consideration"))
                continue
            if head:
                other = re.sub(r"\s+", " ", (d.get("full_text") or "")[:1200]).strip().lower()
                sm = difflib.SequenceMatcher(None, head, other)
                # quick_ratio is an upper bound: only pay for the real ratio if it could pass
                if sm.quick_ratio() >= self.self_similarity_threshold and \
                        sm.ratio() >= self.self_similarity_threshold:
                    excluded.append((d, f"text overlap {sm.ratio():.2f} with the case under consideration"))
                    continue
            pool.append(d)
        return pool, excluded

    def _retrieve_candidates(self, pool: list[dict], query: str):
        """Hybrid BM25 + dense retrieval, with a BM25-only fallback for tiny corpora."""
        k = min(self.candidate_pool, len(pool))
        if len(pool) >= 2:
            try:
                from retrieval.hybrid import HybridRetriever
                retriever = HybridRetriever(pool)
                results = retriever.search(query, top_k=k)
                by_id = {d["judgment_id"]: d for d in pool}
                ranked = [(by_id[r["judgment_id"]],
                           {"hybrid": r["hybrid_score"], "bm25": r["bm25_score"], "dense": r["dense_score"]})
                          for r in results if r["judgment_id"] in by_id]
                if ranked:
                    return ranked, "hybrid"
            except Exception:  # noqa: BLE001 - e.g. SVD cannot fit a 2-doc corpus
                pass

        from retrieval.sparse.bm25 import BM25
        bm25 = BM25([d["full_text"] for d in pool])
        raw = bm25.rank(query, top_k=k)
        top = max((s for _, s in raw), default=0.0) or 1.0
        ranked = [(pool[i], {"hybrid": round(s / top, 4), "bm25": round(s, 4), "dense": 0.0})
                  for i, s in raw]
        return ranked, "bm25_only"

    @staticmethod
    def _query_profile(fact_sheet, user_query, case_text):
        """The pending case's fact-pattern signals, statutory hooks and category."""
        if fact_sheet is not None:
            corpus = " ".join(f.text for f in (fact_sheet.facts + fact_sheet.claims
                                               + fact_sheet.defences + fact_sheet.procedural_points))
            hooks = {h["concept"] for h in fact_sheet.legal_hooks}
            category = fact_sheet.dispute_category
        else:
            corpus, hooks, category = case_text, set(), None
        signals = case_signals(f"{corpus} {user_query}")
        return signals, hooks, category

    def _build_match(self, doc, scores, query_signals, query_hooks, query_category, fact_sheet) -> PrecedentMatch:
        text = doc.get("full_text") or ""
        forum = doc.get("forum") or detect_forum_extended(text)

        from agents.fact_extraction import detect_category, CPA_HOOKS
        cand_category, _ = detect_category(text)
        cand_signals = case_signals(text)
        cand_hooks = {c for c, meta in CPA_HOOKS.items() if re.search(meta["cues"], text, re.IGNORECASE)}

        # --- the six re-ranking signals, each in [0, 1]
        comp = {
            "retrieval": float(scores.get("hybrid", 0.0)),
            "category": 1.0 if (query_category and cand_category == query_category) else 0.0,
            "hooks": jaccard(query_hooks, cand_hooks),
            "signals": jaccard(query_signals, cand_signals),
            "authority": authority_weight(forum),
            "recency": recency_score(doc.get("date")),
        }
        final = sum(SCORING_WEIGHTS[k] * v for k, v in comp.items())

        m = PrecedentMatch(
            judgment_id=doc.get("judgment_id"),
            case_number=doc.get("case_number"),
            forum=forum,
            date=doc.get("date"),
            final_score=round(final, 4),
            score_components={k: round(v, 3) for k, v in comp.items()},
            retrieval_scores=scores,
            dispute_category=cand_category,
            shared_hooks=sorted(query_hooks & cand_hooks),
            shared_signals=sorted(query_signals & cand_signals),
            distinguishing_factors=self._distinguish(query_signals, cand_signals, query_category,
                                                     cand_category, fact_sheet, text),
            holdings=extract_holdings(text, self.max_holdings),
            outcome=extract_outcome(text),
            snippet=re.sub(r"\s+", " ", text[:250]).strip() + "...",
        )
        m.why_relevant = self._explain(m, comp)
        return m

    @staticmethod
    def _distinguish(query_signals, cand_signals, query_category, cand_category,
                     fact_sheet, cand_text) -> list[str]:
        """
        Where the precedent's facts diverge from the pending case.

        Argument Analysis needs this as much as the similarity: an opposing
        counsel distinguishes a precedent on exactly these points.
        """
        out = []
        if query_category and cand_category and cand_category != query_category:
            out.append(f"different dispute category ({cand_category} vs {query_category})")
        for s in sorted(query_signals - cand_signals):
            out.append(f"pending case has '{s}', precedent does not")
        for s in sorted(cand_signals - query_signals):
            out.append(f"precedent turns on '{s}', absent here")

        # amount bracket: a Rs.70 lakh flat and a Rs.45,000 fridge are not alike
        if fact_sheet is not None and fact_sheet.total_amount_paid_inr:
            cand_amounts = parse_amounts(cand_text)
            if cand_amounts:
                qa, ca = fact_sheet.total_amount_paid_inr, max(cand_amounts)
                if qa > 0 and ca > 0 and abs(math.log10(ca / qa)) >= 1:
                    out.append(f"amount in issue differs by an order of magnitude (Rs.{ca:,} vs Rs.{qa:,})")
        return out[:6]

    @staticmethod
    def _explain(m: PrecedentMatch, comp: dict) -> list[str]:
        """Human-readable reasons, one per contributing signal. Nothing unscored."""
        why = []
        if comp["retrieval"] > 0:
            why.append(f"text relevance to the precedent query ({comp['retrieval']:.2f})")
        if comp["category"]:
            why.append(f"same dispute category ({m.dispute_category})")
        if m.shared_hooks:
            why.append("shares statutory basis: " + ", ".join(m.shared_hooks))
        if m.shared_signals:
            why.append("shares fact pattern: " + ", ".join(m.shared_signals))
        why.append(f"{m.forum or 'unknown forum'} carries authority weight {comp['authority']:.2f}")
        if m.date:
            why.append(f"decided {m.date} (recency {comp['recency']:.2f})")
        if m.outcome.label != "unknown":
            why.append(f"operative outcome '{m.outcome.label}' favours the {m.outcome.favours}")
        return why

    def _llm_rescore(self, shortlist: list[PrecedentMatch], fact_sheet, bundle: PrecedentBundle):
        """
        Blend an LLM relevance judgement into the rule score (50/50).

        The LLM only re-scores candidates retrieval already found in the
        authenticated corpus, so it can re-order the shortlist but cannot
        introduce a case - the hallucination surface is zero by construction.
        """
        try:
            case_summary = (fact_sheet.summary()[:2500] if fact_sheet
                            else f"Query: {bundle.precedent_query}")
            listing = "\n".join(
                f"{i}: {m.case_number} ({m.forum}, {m.date}) outcome={m.outcome.label} "
                f"signals={','.join(m.shared_signals) or 'none'} :: {m.snippet[:200]}"
                for i, m in enumerate(shortlist))
            reply = self.llm.complete_json(
                LLM_SYSTEM, f"PENDING CASE\n{case_summary}\n\nCANDIDATE PRECEDENTS\n{listing}")
            for item in reply:
                i = int(item["id"])
                if not 0 <= i < len(shortlist):
                    continue
                rel = max(0.0, min(1.0, float(item.get("relevance", 0.0))))
                m = shortlist[i]
                m.score_components["llm_relevance"] = round(rel, 3)
                m.final_score = round(0.5 * m.final_score + 0.5 * rel, 4)
                m.llm_rationale = str(item.get("reason", ""))[:300]
                m.ranked_by = "rules+llm"
        except Exception as e:  # noqa: BLE001 - any failure => keep the rule ranking
            self.llm_error = str(e)
            bundle.warnings.append(f"LLM re-ranking unavailable ({e}); using rule-based ranking.")

    def _verify(self, m: PrecedentMatch, conn):
        """
        CiteVerify gate. A precedent the system cannot authenticate against the
        curated corpus is still shown, flagged UNVERIFIED with a reason - but
        `admissible_as_authority` stays False, so Prediction and Explanation
        can refuse to rely on it.
        """
        target = m.case_number or m.snippet[:80]
        verdict = verify_precedent(target, conn, similarity_threshold=self.verify_threshold)
        m.verified = verdict.verified
        m.verification_confidence = verdict.confidence
        m.verification_reason = verdict.reason
        m.admissible_as_authority = bool(verdict.verified and m.holdings)
        if verdict.verified and not m.holdings:
            m.verification_reason += " (no holding sentence located; not usable as authority)"

    def _attach_cited_authorities(self, m: PrecedentMatch, conn):
        """
        Citation-graph expansion: the cases a retrieved precedent itself relies
        on are candidate authorities too. Each is verified before it is passed
        on - most will be outside the ingested corpus and correctly report as
        UNVERIFIED rather than being presented as checked law.
        """
        if m.judgment_id is None:
            return
        try:
            rows = conn.execute(
                "SELECT case_name, year, reporter_citation FROM precedent_citations WHERE judgment_id = ?",
                (m.judgment_id,)).fetchall()
        except Exception:  # noqa: BLE001 - table absent in a bare DB
            return
        for name, year, reporter in rows[:5]:
            v = verify_precedent(name, conn, similarity_threshold=self.verify_threshold)
            m.cited_authorities.append(CitedAuthority(
                case_name=name, year=year, reporter_citation=reporter,
                cited_in_judgment_id=m.judgment_id,
                verified=v.verified, verification_confidence=v.confidence,
                verification_reason=v.reason))

    @staticmethod
    def _aggregate_evidence(bundle: PrecedentBundle):
        """
        The prior the Prediction Agent conditions on.

        Two rates are reported: a plain count, and one weighted by forum
        authority, because three District Forum orders should not outweigh one
        NCDRC decision. Only verified matches count toward either.
        """
        usable = [m for m in bundle.matches if m.verified and m.outcome.label != "unknown"]
        dist: dict[str, int] = {}
        for m in bundle.matches:
            dist[m.outcome.label] = dist.get(m.outcome.label, 0) + 1
        bundle.outcome_distribution = dist
        if not usable:
            return
        favourable = [m for m in usable if m.outcome.favours == "complainant"]
        bundle.complainant_success_rate = round(len(favourable) / len(usable), 3)
        total_w = sum(authority_weight(m.forum) for m in usable)
        if total_w:
            bundle.authority_weighted_success_rate = round(
                sum(authority_weight(m.forum) for m in favourable) / total_w, 3)
