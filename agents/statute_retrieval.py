"""
Agent 2 of the Multi-Agent Legal Reasoning Pipeline: STATUTE RETRIEVAL AGENT.

Objective (project synopsis): retrieve the legal evidence - here the statutory
basis - for the case under consideration, so that Precedent Retrieval,
Argument Analysis and Prediction reason against identified provisions rather
than against an unstated notion of "the law".

Position in the pipeline
------------------------
    Fact Extraction -> [STATUTE RETRIEVAL] -> Precedent Retrieval ->
    Argument Analysis -> Prediction -> Verification -> Explanation

It consumes the `fact_sheet` written by Agent 1 and writes `statutes` into the
shared state - exactly the key the Precedent Retrieval Agent already reads
(`state["statutes"]`, entries with `section` and `act`), so adding this agent
strengthens Agent 3 without changing it.

What it does
------------
1. Decides WHICH ACT applies, from the case's reference date. The 2019 Act's
   consumer-dispute provisions commenced on 20 July 2020; a 2016 dispute is
   governed by the 1986 Act. Citing s.2(11) at a pre-2020 cause of action is a
   straightforward legal error, so this is settled before anything is retrieved
   (`statutes/cpa_corpus.py: act_in_force`).
2. Retrieves candidate provisions through four independent channels, each
   recorded as provenance on the result:
      - concept hooks emitted by Agent 1 (deterministic mapping)
      - element matching against the fact sheet
      - BM25 lexical retrieval over the provision corpus
      - corpus evidence: how often the provision is actually cited by
        judgments already ingested into the structured legal DB
   Each candidate is then weighted by its ROLE in a complaint: a cause of
   action outranks a definition, which outranks a remedy, a limitation
   defence, or a jurisdiction provision. Without this a builder-delay case
   returns s.69 (limitation) as its leading provision, because the word
   "limitation" is lexically prominent in a judgment that rejects that very
   defence.
3. ELEMENT MATCHING is the substance of the agent. Each provision carries the
   elements a complainant must establish; the agent reports, per element,
   which fact IDs satisfy it and which elements are unsatisfied. A section
   number alone is an assertion; a section number with "element 2 is satisfied
   by F3 and F7, element 3 is unsatisfied" is an argument the Argument
   Analysis Agent can actually use and a reader can check.
4. Applies two statutory tests that turn on computable facts:
      - pecuniary jurisdiction, from the amount in issue and the bands in
        force on the relevant date
      - limitation (s.69 / s.24A), from the cause-of-action date and the
        reference date, flagged as indeterminate rather than guessed when the
        dates are not in the fact sheet
5. Separates provisions that help the complainant from those the opposite
   party relies on (limitation, jurisdiction), so Argument Analysis receives
   both sides.
6. Verification-first (Objective 4): every provision is put through
   CiteVerify's `verify_statute` before it can be relied on. An unverified
   provision is surfaced flagged, never silently dropped and never silently
   passed.

Honesty note
------------
The provision corpus holds plain-language summaries, not gazette text
(`verbatim=False` on every entry). The agent propagates that flag to every
match and to the bundle, so no downstream agent can mistake a retrieval aid
for authenticated statutory text. See `statutes/cpa_corpus.py`.

Rules vs. LLM
-------------
Offline and rule-based by default (same trade-off as Agents 1 and 3). Pass
`llm=AnthropicBackend()` to have an LLM re-score applicability; the LLM only
re-scores provisions already retrieved from the curated corpus, so it can
re-order but cannot invent a section. Any failure falls back to the rules.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from datetime import date

from agents.base import BaseAgent
from agents.fact_extraction import CPA_HOOKS, FactSheet
from statutes.cpa_corpus import (
    ALL_PROVISIONS, ROLE_PRIOR, CPA_1986, CPA_2019, CPA_2019_COMMENCEMENT, LIMITATION_YEARS,
    PROVISIONS_BY_ACT, act_in_force, forum_for_value, get_provision,
    pecuniary_regime, provision_text,
)
from verification.citeverify import verify_statute

# --------------------------------------------------------------------------
# Scoring. Exposed so the review panel (and the ablation table) can vary it.
# --------------------------------------------------------------------------
SCORING_WEIGHTS = {
    "elements": 0.30,     # fraction of the provision's elements the facts satisfy
    "role": 0.20,         # the provision's role in a complaint (see ROLE_PRIOR)
    "hooks": 0.20,        # concept hooks Agent 1 emitted, weighted by cue strength
    "retrieval": 0.12,    # BM25 over the provision corpus
    "corpus_support": 0.10,  # how often ingested judgments cite this provision
    "applicability": 0.08,   # provision belongs to the Act actually in force
}

# A provision from the wrong Act is not merely less relevant, it is wrong.
# It is kept (so the agent can show the equivalent) but heavily discounted.
WRONG_ACT_PENALTY = 0.35

CAUSE_OF_ACTION_LABELS = ("deficiency",)  # fact labels that mark when the wrong occurred

# Language indicating a continuing wrong, where tribunals have treated the
# cause of action as recurring rather than accruing once.
RECURRING_CAUSE = re.compile(
    r"\b(till date|to date|even now|never (delivered|offered|handed)|"
    r"has not been (delivered|offered|handed|completed)|recurrent cause of action|"
    r"continuing cause of action)\b", re.IGNORECASE)


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------
@dataclass
class ElementFinding:
    """One statutory element, and the facts that do or do not satisfy it."""
    element: str
    satisfied: bool
    fact_ids: list = field(default_factory=list)
    evidence: str = ""


@dataclass
class StatuteMatch:
    section: str
    act: str
    title: str
    summary: str
    role: str = "definition"
    final_score: float = 0.0
    score_components: dict = field(default_factory=dict)
    elements: list = field(default_factory=list)          # list[ElementFinding]
    element_coverage: float = 0.0
    unsatisfied_elements: list = field(default_factory=list)
    concepts: list = field(default_factory=list)
    provenance: list = field(default_factory=list)        # which channels found it
    favours: str = "complainant"
    in_force_for_this_case: bool = True
    equivalent_provision: str | None = None
    corpus_citations: int = 0
    verified: bool = False
    verification_confidence: float = 0.0
    verification_reason: str = ""
    verbatim_text_available: bool = False
    numbering_verified: bool = True
    admissible_as_authority: bool = False
    why_relevant: list = field(default_factory=list)
    notes: str = ""
    ranked_by: str = "rules"
    llm_rationale: str | None = None

    @property
    def citation(self) -> str:
        return f"Section {self.section} of the {self.act}"


@dataclass
class JurisdictionFinding:
    amount_in_issue_inr: int | None = None
    forum: str | None = None
    regime: str | None = None
    basis: str = ""
    inputs: list = field(default_factory=list)
    determinate: bool = False
    note: str = ""


@dataclass
class LimitationFinding:
    status: str = "indeterminate"      # within_time | prima_facie_barred | indeterminate
    cause_of_action_date: str | None = None
    reference_date: str | None = None
    years_elapsed: float | None = None
    limit_years: int = LIMITATION_YEARS
    provision: str | None = None
    inputs: list = field(default_factory=list)
    continuing_cause_indicated: bool = False
    note: str = ""


@dataclass
class StatuteBundle:
    statute_query: str = ""
    governing_act: str = CPA_2019
    governing_act_reason: str = ""
    reference_date: str | None = None
    matches: list = field(default_factory=list)           # list[StatuteMatch]
    complainant_provisions: list = field(default_factory=list)   # citations
    opposite_party_provisions: list = field(default_factory=list)
    jurisdiction: JurisdictionFinding = field(default_factory=JurisdictionFinding)
    limitation: LimitationFinding = field(default_factory=LimitationFinding)
    verified_count: int = 0
    unverified_count: int = 0
    verbatim_text_available: bool = False
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def for_precedent_agent(self) -> list[dict]:
        """The `state["statutes"]` contract the Precedent Retrieval Agent reads."""
        return [{"section": m.section, "act": m.act, "title": m.title,
                 "score": m.final_score, "favours": m.favours, "verified": m.verified}
                for m in self.matches]

    def summary(self) -> str:
        L = [f"Governing Act: {self.governing_act}",
             f"  reason: {self.governing_act_reason}",
             f"Statute query: {self.statute_query}",
             f"Verified against curated provisions: {self.verified_count}  |  "
             f"unverified: {self.unverified_count}",
             f"Gazette text loaded: {self.verbatim_text_available} "
             f"(summaries are retrieval aids, not statutory text)"]
        L.append(f"\nAPPLICABLE PROVISIONS ({len(self.matches)})")
        for i, m in enumerate(self.matches, 1):
            flag = "VERIFIED" if m.verified else "UNVERIFIED"
            force = "" if m.in_force_for_this_case else "  [NOT THE ACT IN FORCE FOR THIS CASE]"
            L.append(f"\n{i}. [{m.final_score:.3f}] {m.citation}  [{flag}]{force}")
            L.append(f"   {m.title} - role: {m.role}, favours {m.favours}")
            L.append("   components: " + "  ".join(f"{k}={v:.2f}" for k, v in m.score_components.items()))
            L.append(f"   element coverage: {m.element_coverage:.0%}")
            for e in m.elements:
                mark = "satisfied" if e.satisfied else "NOT satisfied"
                ids = f" by {', '.join(e.fact_ids)}" if e.fact_ids else ""
                L.append(f"     - {e.element}: {mark}{ids}")
                if e.evidence:
                    L.append(f"       {e.evidence[:140]}")
            if m.corpus_citations:
                L.append(f"   cited by {m.corpus_citations} ingested judgment(s)")
            if m.equivalent_provision:
                L.append(f"   equivalent under the other Act: {m.equivalent_provision}")
            if not m.numbering_verified:
                L.append("   section numbering NOT verified against the gazette - do not cite as-is")
            if m.notes:
                L.append(f"   note: {m.notes}")
            if m.llm_rationale:
                L.append(f"   llm: {m.llm_rationale}")

        j = self.jurisdiction
        L.append("\nJURISDICTION (s.34/47/58)")
        if j.determinate:
            L.append(f"  amount in issue Rs.{j.amount_in_issue_inr:,} -> {j.forum}")
            L.append(f"  bands applied: {j.regime} ({j.basis})")
            L.append(f"  inputs: {', '.join(j.inputs)}")
        else:
            L.append(f"  indeterminate - {j.note}")

        lim = self.limitation
        L.append(f"\nLIMITATION ({lim.provision or 's.69 / s.24A'})")
        L.append(f"  status: {lim.status}")
        if lim.years_elapsed is not None:
            L.append(f"  cause of action {lim.cause_of_action_date} -> reference {lim.reference_date} "
                     f"= {lim.years_elapsed:.1f} years against a {lim.limit_years}-year limit")
        if lim.inputs:
            L.append(f"  inputs: {', '.join(lim.inputs)}")
        if lim.note:
            L.append(f"  note: {lim.note}")

        L.append("\nFOR ARGUMENT ANALYSIS")
        L.append(f"  complainant relies on: {', '.join(self.complainant_provisions) or '-'}")
        L.append(f"  opposite party relies on: {', '.join(self.opposite_party_provisions) or '-'}")
        for w in self.warnings:
            L.append(f"WARNING: {w}")
        return "\n".join(L)


# --------------------------------------------------------------------------
# Helper functions (pure, individually testable)
# --------------------------------------------------------------------------
def fact_corpus(fact_sheet: FactSheet | None, extra: str = "") -> list[tuple[str, str]]:
    """[(fact_id, text)] over every classified sentence Agent 1 produced."""
    if fact_sheet is None:
        return [("Q1", extra)] if extra.strip() else []
    items = [(f.id, f.text) for f in (fact_sheet.facts + fact_sheet.claims
                                      + fact_sheet.defences + fact_sheet.procedural_points)]
    if extra.strip():
        items.append(("Q1", extra))
    return items


def match_elements(provision: dict, corpus: list[tuple[str, str]]) -> list[ElementFinding]:
    """
    Test each statutory element against the facts.

    Grounded by construction: an element is satisfied only by naming the fact
    IDs that satisfy it, so a reader can go back to Agent 1's offsets and read
    the sentence in the source.
    """
    findings = []
    for name, pattern in provision["elements"]:
        rx = re.compile(pattern, re.IGNORECASE)
        hits = [(fid, text) for fid, text in corpus if rx.search(text)]
        findings.append(ElementFinding(
            element=name,
            satisfied=bool(hits),
            fact_ids=[fid for fid, _ in hits][:6],
            evidence=hits[0][1] if hits else "",
        ))
    return findings


def build_statute_query(fact_sheet: FactSheet | None, user_query: str = "") -> str:
    """Statutory concepts first, then the dispute category, then the user's wording."""
    parts: list[str] = []
    if fact_sheet is not None:
        parts += [h["concept"].replace("_", " ") for h in fact_sheet.legal_hooks]
        if fact_sheet.dispute_category:
            parts.append(fact_sheet.dispute_category.replace("_", " "))
        for f in (fact_sheet.claims + fact_sheet.facts)[:12]:
            parts.append(f.text)
    if user_query.strip():
        parts.append(user_query.strip())
    tokens = " ".join(parts).split()
    return " ".join(dict.fromkeys(tokens))[:1200]


def corpus_citation_counts(conn, act_hint: str | None = None) -> dict[tuple[str, str], int]:
    """
    How often each provision is cited by judgments already ingested.

    A provision the ingested tribunals actually rely on for this kind of
    dispute is better evidence of applicability than lexical similarity.
    """
    counts: dict[tuple[str, str], int] = {}
    try:
        rows = conn.execute("SELECT section, act, COUNT(*) FROM statute_citations "
                            "GROUP BY section, act").fetchall()
    except Exception:  # noqa: BLE001 - table absent in a bare DB
        return counts
    for section, act, n in rows:
        if not section:
            continue
        key = (section.strip(), (act or act_hint or "").strip())
        counts[key] = counts.get(key, 0) + n
    return counts


def cause_of_action_date(fact_sheet: FactSheet | None) -> tuple[str | None, list[str]]:
    """
    The earliest date attached to a deficiency fact - when the wrong occurred.

    Deliberately narrow: the transaction date is when the contract was made,
    not when it was breached, and using it would make every complaint look
    time-barred.
    """
    if fact_sheet is None:
        return None, []
    candidates = [(d, f.id) for f in fact_sheet.facts
                  if f.label in CAUSE_OF_ACTION_LABELS for d in f.dates]
    if not candidates:
        # a computed deadline is the next best marker of when the breach began
        for d in fact_sheet.derived_facts:
            if d.name == "promised_delivery_deadline":
                return d.value, list(d.inputs)
        return None, []
    candidates.sort()
    return candidates[0][0], [candidates[0][1]]


# --------------------------------------------------------------------------
# The agent
# --------------------------------------------------------------------------
LLM_SYSTEM = (
    "You are a legal research assistant for Indian consumer-dispute adjudication. "
    "You are given a structured fact sheet of a pending case and a numbered shortlist of "
    "statutory provisions that a retrieval system already found in a curated corpus of the "
    "Consumer Protection Act. Score how directly each provision governs the pending case. "
    'Respond ONLY with JSON: [{"id": <int>, "applicability": <0-1>, "reason": "<one short sentence>"}]. '
    "Score only the provisions given. Never add a section, an Act or a holding that is not in the list."
)


class StatuteRetrievalAgent(BaseAgent):
    """
    Agent 2: identifies the statutory basis of the pending case.

    Usage inside the pipeline (LangGraph-compatible, same contract as Agents 1 and 3):
        state = FactExtractionAgent().run(state)
        state = StatuteRetrievalAgent(conn=conn).run(state)      # writes state["statutes"]
        state = PrecedentRetrievalAgent(conn=conn).run(state)    # reads state["statutes"]
    """
    name = "statute_retrieval_agent"

    def __init__(self, conn=None, db_path: str = "data_pipeline/legal.db",
                 top_k: int = 6, llm=None, min_score: float = 0.15,
                 include_other_act: bool = False):
        """
        conn: sqlite3 connection to the structured legal DB, used only for the
              corpus-support signal. If None it is opened lazily; if the DB is
              missing the agent still runs, minus that one signal.
        include_other_act: also return provisions from the Act that is NOT in
              force for this case (discounted, flagged). Useful for showing the
              1986 equivalent alongside a 2019 provision.
        """
        self.conn = conn
        self.db_path = db_path
        self.top_k = top_k
        self.llm = llm
        self.min_score = min_score
        self.include_other_act = include_other_act
        self.llm_error: str | None = None

    # ---- pipeline interface -------------------------------------------------
    def run(self, state: dict) -> dict:
        bundle = self.retrieve(
            fact_sheet=state.get("fact_sheet"),
            case_text=state.get("case_text", ""),
            user_query=state.get("query", ""),
        )
        state["statute_bundle"] = bundle
        state["statutes"] = bundle.for_precedent_agent()
        state["statute_query"] = bundle.statute_query
        return state

    # ---- main entry ---------------------------------------------------------
    def retrieve(self, fact_sheet: FactSheet | None = None, case_text: str = "",
                 user_query: str = "") -> StatuteBundle:
        bundle = StatuteBundle()
        bundle.statute_query = build_statute_query(fact_sheet, user_query)

        # --- 1. which Act governs
        ref = self._reference_date(fact_sheet)
        bundle.reference_date = ref.isoformat() if ref else None
        bundle.governing_act, bundle.governing_act_reason = act_in_force(ref)
        if ref is None:
            bundle.warnings.append(
                "No reference date in the fact sheet - the Act in force was assumed, not determined. "
                "A pre-2020 cause of action is governed by the 1986 Act.")

        corpus = fact_corpus(fact_sheet, case_text or user_query)
        if not corpus:
            bundle.warnings.append("No facts and no case text supplied - nothing to match provisions against.")
            return bundle

        # --- 2. candidate scoring across four channels
        # Hook strength, not hook presence. Agent 1 reports how many cues fired for
        # each concept; a case with 14 deficiency cues and 2 limitation cues is a
        # deficiency case with a limitation defence, and the ranking must say so.
        # Binary hook matching scores both at 1.0 and floats the defence to the top.
        hooks = {h["concept"]: h.get("cue_hits", 1) for h in fact_sheet.legal_hooks} if fact_sheet else {}
        counts = corpus_citation_counts(self._get_conn_safe(), bundle.governing_act)
        max_count = max(counts.values(), default=0)
        retrieval = self._bm25_scores(bundle.statute_query)

        pool = (PROVISIONS_BY_ACT[bundle.governing_act] if not self.include_other_act
                else ALL_PROVISIONS)
        max_hits = max(hooks.values(), default=0)
        matches = [self._build_match(p, corpus, hooks, max_hits, retrieval, counts, max_count, bundle)
                   for p in pool]

        # --- 3. optional LLM re-scoring of the shortlist (never adds a section)
        matches.sort(key=lambda m: -m.final_score)
        shortlist = matches[:max(self.top_k * 2, 8)]
        if self.llm is not None:
            self._llm_rescore(shortlist, fact_sheet, bundle)

        selected = [m for m in shortlist if m.final_score >= self.min_score]
        selected.sort(key=lambda m: -m.final_score)
        selected = selected[:self.top_k]

        # --- 4. verification-first
        for m in selected:
            self._verify(m)

        bundle.matches = selected
        bundle.verified_count = sum(1 for m in selected if m.verified)
        bundle.unverified_count = len(selected) - bundle.verified_count
        bundle.complainant_provisions = [m.citation for m in selected if m.favours == "complainant"]
        bundle.opposite_party_provisions = [m.citation for m in selected if m.favours == "opposite_party"]

        # --- 5. the two computable statutory tests
        bundle.jurisdiction = self._jurisdiction(fact_sheet, ref, bundle)
        bundle.limitation = self._limitation(fact_sheet, ref, bundle)

        if not selected:
            bundle.warnings.append(
                f"No provision scored at or above min_score={self.min_score}; the fact sheet may be "
                "too thin to establish a statutory basis.")
        if not any(m.verified for m in selected):
            bundle.warnings.append(
                "No provision could be authenticated against the curated provisions list - "
                "CiteVerify's registry may be missing the sub-clauses these matches use.")
        return bundle

    # ---- internals ----------------------------------------------------------
    def _get_conn_safe(self):
        if self.conn is not None:
            return self.conn
        try:
            from structuring.db import get_conn
            self.conn = get_conn(self.db_path)
        except Exception:  # noqa: BLE001 - corpus support is optional, not required
            self.conn = None
        return self.conn

    @staticmethod
    def _reference_date(fact_sheet: FactSheet | None) -> date | None:
        if fact_sheet is None or not fact_sheet.reference_date:
            return None
        try:
            return date.fromisoformat(fact_sheet.reference_date)
        except ValueError:
            return None

    @staticmethod
    def _bm25_scores(query: str) -> dict[tuple[str, str], float]:
        """BM25 over the provision corpus, min-max normalised to [0, 1]."""
        if not query.strip():
            return {}
        from retrieval.sparse.bm25 import BM25
        keys = [(p["section"], p["act"]) for p in ALL_PROVISIONS]
        bm25 = BM25([provision_text(p) for p in ALL_PROVISIONS])
        raw = dict(bm25.rank(query, top_k=len(ALL_PROVISIONS)))
        top = max(raw.values(), default=0.0) or 1.0
        return {keys[i]: s / top for i, s in raw.items()}

    def _build_match(self, p: dict, corpus, hooks, max_hits, retrieval, counts, max_count,
                     bundle: StatuteBundle) -> StatuteMatch:
        findings = match_elements(p, corpus)
        coverage = (sum(1 for f in findings if f.satisfied) / len(findings)) if findings else 0.0
        hook_overlap = max((hooks[c] / max_hits for c in p["concepts"] if c in hooks),
                           default=0.0) if max_hits else 0.0
        n_cites = counts.get((p["section"], p["act"]), 0) or counts.get((p["section"], ""), 0)
        in_force = p["act"] == bundle.governing_act

        comp = {
            "elements": coverage,
            "hooks": hook_overlap,
            "retrieval": retrieval.get((p["section"], p["act"]), 0.0),
            "corpus_support": (n_cites / max_count) if max_count else 0.0,
            "applicability": 1.0 if in_force else 0.0,
            "role": ROLE_PRIOR[p["role"]],
        }
        score = sum(SCORING_WEIGHTS[k] * v for k, v in comp.items())
        if not in_force:
            score *= WRONG_ACT_PENALTY

        eq = p.get("equivalent")
        m = StatuteMatch(
            section=p["section"], act=p["act"], title=p["title"], summary=p["summary"],
            role=p["role"],
            final_score=round(score, 4),
            score_components={k: round(v, 3) for k, v in comp.items()},
            elements=findings,
            element_coverage=round(coverage, 3),
            unsatisfied_elements=[f.element for f in findings if not f.satisfied],
            concepts=list(p["concepts"]),
            provenance=self._provenance(comp),
            favours=p["favours"],
            in_force_for_this_case=in_force,
            equivalent_provision=(f"Section {eq[0]} of the {eq[1]}" if eq else None),
            corpus_citations=n_cites,
            verbatim_text_available=p["verbatim"],
            numbering_verified=p["numbering_verified"],
            notes=p["notes"],
        )
        m.why_relevant = self._explain(m, comp)
        return m

    @staticmethod
    def _provenance(comp) -> list[str]:
        """Which retrieval channel found this provision. Nothing appears unexplained."""
        out = []
        if comp["elements"] > 0:
            out.append("element_matching")
        if comp["hooks"] > 0:
            out.append("concept_hooks")
        if comp["retrieval"] > 0:
            out.append("bm25_provision_corpus")
        if comp["corpus_support"] > 0:
            out.append("cited_by_ingested_judgments")
        return out

    @staticmethod
    def _explain(m: StatuteMatch, comp: dict) -> list[str]:
        why = []
        satisfied = [f for f in m.elements if f.satisfied]
        if satisfied:
            why.append(f"{len(satisfied)} of {len(m.elements)} statutory elements satisfied by "
                       + ", ".join(sorted({fid for f in satisfied for fid in f.fact_ids})[:8]))
        if m.unsatisfied_elements:
            why.append("not yet established: " + "; ".join(m.unsatisfied_elements))
        if comp["hooks"] > 0:
            why.append(f"matches concept hooks ({comp['hooks']:.2f} of the case's strongest hook): "
                       + ", ".join(m.concepts))
        if m.corpus_citations:
            why.append(f"relied on by {m.corpus_citations} ingested judgment(s)")
        if not m.in_force_for_this_case:
            why.append("belongs to the Act NOT in force for this case - shown for comparison only")
        return why

    def _llm_rescore(self, shortlist: list[StatuteMatch], fact_sheet, bundle: StatuteBundle):
        """Blend an LLM applicability judgement 50/50 into the rule score."""
        try:
            case_summary = (fact_sheet.summary()[:2500] if fact_sheet
                            else f"Query: {bundle.statute_query[:500]}")
            listing = "\n".join(
                f"{i}: {m.citation} - {m.title}. {m.summary[:160]} "
                f"(elements satisfied {m.element_coverage:.0%})"
                for i, m in enumerate(shortlist))
            reply = self.llm.complete_json(
                LLM_SYSTEM, f"PENDING CASE\n{case_summary}\n\nCANDIDATE PROVISIONS\n{listing}")
            for item in reply:
                i = int(item["id"])
                if not 0 <= i < len(shortlist):
                    continue
                app = max(0.0, min(1.0, float(item.get("applicability", 0.0))))
                m = shortlist[i]
                m.score_components["llm_applicability"] = round(app, 3)
                m.final_score = round(0.5 * m.final_score + 0.5 * app, 4)
                m.llm_rationale = str(item.get("reason", ""))[:300]
                m.ranked_by = "rules+llm"
        except Exception as e:  # noqa: BLE001 - any failure => keep the rule ranking
            self.llm_error = str(e)
            bundle.warnings.append(f"LLM re-scoring unavailable ({e}); using rule-based ranking.")

    @staticmethod
    def _verify(m: StatuteMatch):
        """
        CiteVerify gate. A provision that cannot be authenticated against the
        curated list is still shown, flagged, with `admissible_as_authority`
        False, so Explanation can refuse to cite it.
        """
        verdict = verify_statute(m.section, m.act)
        m.verified = verdict.verified
        m.verification_confidence = verdict.confidence
        m.verification_reason = verdict.reason
        m.admissible_as_authority = bool(verdict.verified and m.numbering_verified)

    # ---- the two computable statutory tests ---------------------------------
    @staticmethod
    def _jurisdiction(fact_sheet, ref: date | None, bundle: StatuteBundle) -> JurisdictionFinding:
        """
        Pecuniary jurisdiction from the amount in issue.

        Reported as indeterminate, with the reason, when the fact sheet holds
        no amount - a guessed forum is worse than an acknowledged gap.
        """
        f = JurisdictionFinding()
        amount = fact_sheet.total_amount_paid_inr if fact_sheet else None
        if not amount:
            f.note = "no amount in issue found in the fact sheet"
            return f
        claimed = [a for c in (fact_sheet.claims if fact_sheet else []) for a in c.amounts_inr]
        total = amount + (max(claimed) if claimed else 0)
        forum, regime = forum_for_value(total, ref)
        f.amount_in_issue_inr = total
        f.forum = forum
        f.regime = regime["regime"]
        f.basis = regime["basis"]
        f.determinate = True
        f.inputs = ["fact_sheet.total_amount_paid_inr"] + (["largest claimed amount"] if claimed else [])
        if not regime["verified"]:
            bundle.warnings.append(
                f"Pecuniary bands for '{regime['regime']}' are unverified in the provision corpus - "
                "confirm the current jurisdiction rules before relying on this forum finding.")
        return f

    @staticmethod
    def _limitation(fact_sheet, ref: date | None, bundle: StatuteBundle) -> LimitationFinding:
        """
        Limitation under s.69 (2019) / s.24A (1986): two years from the cause
        of action, subject to condonation.

        Where the facts indicate a continuing wrong (possession never
        delivered), the finding is reported with that caveat instead of a bare
        "barred" - tribunals have treated such causes of action as recurring,
        and a flat "time-barred" would mislead the Prediction Agent.
        """
        lim = LimitationFinding()
        lim.provision = "24A" if bundle.governing_act == CPA_1986 else "69"
        lim.reference_date = ref.isoformat() if ref else None
        coa, inputs = cause_of_action_date(fact_sheet)
        lim.cause_of_action_date = coa
        lim.inputs = inputs

        if fact_sheet is not None:
            blob = " ".join(f.text for f in (fact_sheet.facts + fact_sheet.claims))
            lim.continuing_cause_indicated = bool(RECURRING_CAUSE.search(blob))

        if not coa or not ref:
            lim.note = ("cause-of-action date or reference date missing from the fact sheet; "
                        "limitation not computed rather than guessed")
            return lim
        try:
            start = date.fromisoformat(coa)
        except ValueError:
            lim.note = f"cause-of-action date {coa!r} is not a valid date"
            return lim

        years = (ref - start).days / 365.25
        lim.years_elapsed = round(years, 2)
        if years <= LIMITATION_YEARS:
            lim.status = "within_time"
            lim.note = f"filed within the {LIMITATION_YEARS}-year period"
        elif lim.continuing_cause_indicated:
            lim.status = "indeterminate"
            lim.note = ("more than two years since the first breach, but the facts indicate a "
                        "continuing wrong (a recurrent cause of action) - not prima facie barred "
                        "on these facts alone")
        else:
            lim.status = "prima_facie_barred"
            lim.note = ("beyond the two-year period on the dates in the fact sheet; condonation "
                        "for sufficient cause remains open to the complainant")
        return lim
