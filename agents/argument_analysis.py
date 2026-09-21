"""
Agent 4 of the Multi-Agent Legal Reasoning Pipeline: ARGUMENT ANALYSIS AGENT.

Objective (project synopsis): analyse the arguments - take the facts, the
statutory basis and the retrieved precedents and work out what each side is
actually contending, which contentions are supported, which are rebutted, and
where each issue stands, so that Prediction reasons over a structured argument
map rather than over an undifferentiated pile of evidence.

Position in the pipeline
------------------------
    Fact Extraction -> Statute Retrieval -> Precedent Retrieval ->
    [ARGUMENT ANALYSIS] -> Prediction -> Verification -> Explanation

It consumes `fact_sheet` (Agent 1), `statute_bundle` (Agent 2) and
`precedent_bundle` (Agent 3), and writes `argument_map` into the shared state.
All three inputs are optional: with fewer of them the agent produces a thinner
map and says so in `warnings`, rather than failing.

What it does
------------
1. FRAMES THE ISSUES. A consumer complaint is not one question but several,
   and they are not equal. Threshold issues (is the complainant a consumer, is
   the forum competent, is the complaint in time) can dispose of the case
   whatever the merits; merits issues decide liability; relief issues decide
   quantum. Issues are raised only when something in the case raises them -
   from the statutory provisions Agent 2 returned and the defences Agent 1
   extracted - so the map reflects this case, not a checklist.
2. ASSIGNS EACH CONTENTION TO AN ISSUE. Agent 1 labelled sentences as `claim`
   or `defence`; this agent routes them to the issue they go to, keeping the
   fact ID so every contention remains traceable to a verbatim sentence at a
   known offset.
3. ATTACHES SUPPORT: the statutory provisions and the precedent holdings that
   bear on each issue, for each side. Precedents are attached to the side
   their operative outcome favours, which is the legally meaningful direction -
   a dismissed complaint on the same facts is authority for the opposite party.
4. DETECTS REBUTTAL. A contention the other side never answers is not merely
   unopposed rhetoric: an uncontroverted averment carries weight in consumer
   adjudication. The agent reports, per issue, which contentions are met and
   which stand unanswered.
5. SCORES EACH SIDE on four transparent components - factual grounding,
   statutory support, precedential support, and whether the contentions were
   rebutted - and reports the leaning of the issue with a margin.
6. FLAGS DISPOSITIVE RISK. If a threshold issue leans to the opposite party,
   that can end the case irrespective of the merits, so it is surfaced
   separately rather than being averaged away into an overall score.
7. REPORTS EVIDENTIAL GAPS: statutory elements Agent 2 found unsatisfied, and
   issues where a side has contentions but no support. These are what the
   Explanation Agent should concede and what a user should be told to fix.

Verification-first (Objective 4)
--------------------------------
Only support that passed CiteVerify may carry weight. An unverified provision
or an inadmissible precedent is still attached to the issue - visibly, with a
reason - but it is excluded from the strength scores, so an argument can never
be scored on authority the system could not authenticate.

Leakage
-------
The agent reads only the fact sheet (outcome already stripped by Agent 1) and
precedents (the case under consideration already excluded by Agent 3). It does
not read the judgment text, so it cannot see the result it is helping predict.

Rules vs. LLM
-------------
Offline and rule-based by default (same trade-off as Agents 1-3). Pass
`llm=AnthropicBackend()` to have an LLM re-assign contentions to issues; it may
only re-label contentions that Agent 1 already extracted from the source text,
so it cannot invent a contention. Any failure falls back to the rules.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

from agents.base import BaseAgent
from agents.precedent_retrieval import authority_weight

# --------------------------------------------------------------------------
# Issue taxonomy.
#
# `kind` decides how an issue is treated downstream:
#   threshold - can dispose of the case on its own, whatever the merits
#   merits    - decides liability
#   relief    - decides what follows from liability
# `burden` records who must establish the issue, which is why a weakly
# supported threshold issue hurts the complainant and a weakly supported
# limitation plea hurts the opposite party.
# --------------------------------------------------------------------------
ISSUE_TEMPLATES: dict[str, dict] = {
    "consumer_status": {
        "question": "Whether the complainant is a 'consumer' within the meaning of the Act",
        "kind": "threshold",
        "burden": "complainant",
        "sections": {"2(7)", "2(6)", "2(1)(d)", "2(1)(b)"},
        "concepts": set(),
        "cues": re.compile(
            r"\b(consumer|commercial purpose|resale|for (his|her|their) own use|locus standi|"
            r"not a consumer|beneficiary)\b", re.IGNORECASE),
    },
    "jurisdiction": {
        "question": "Whether this Commission has jurisdiction to entertain the complaint",
        "kind": "threshold",
        "burden": "complainant",
        "sections": {"34", "47", "58", "11", "17", "21"},
        "concepts": {"pecuniary_jurisdiction_issue"},
        "cues": re.compile(
            r"\b(pecuniary jurisdiction|territorial jurisdiction|jurisdiction|maintainab\w+|"
            r"not maintainable|carries on business|cause of action arose)\b", re.IGNORECASE),
    },
    "limitation": {
        "question": "Whether the complaint is barred by limitation",
        "kind": "threshold",
        "burden": "complainant",
        "sections": {"69", "24A"},
        "concepts": {"limitation_issue"},
        "cues": re.compile(
            r"\b(limitation|time[- ]barred|barred by|condon\w+|sufficient cause|"
            r"recurrent cause of action|continuing cause of action|delay in filing)\b", re.IGNORECASE),
    },
    "deficiency": {
        "question": "Whether there was deficiency in service on the part of the opposite party",
        "kind": "merits",
        "burden": "complainant",
        "sections": {"2(11)", "2(42)", "2(1)(g)", "2(1)(o)"},
        "concepts": {"deficiency_in_service"},
        "cues": re.compile(
            r"\b(deficien\w+|delay\w*|failed to|failure to|negligen\w+|not (complete|completed|"
            r"delivered|offered|handed|refunded)|did not (deliver|offer|respond|refund|repair|replace)|"
            r"force majeure|beyond (its|their) control)\b", re.IGNORECASE),
    },
    "defect": {
        "question": "Whether the goods supplied suffered from a defect",
        "kind": "merits",
        "burden": "complainant",
        "sections": {"2(10)", "2(1)(f)", "Chapter VI (ss. 82-87)"},
        "concepts": {"defect_in_goods"},
        "cues": re.compile(
            r"\b(defect(ive|s)?|manufactur\w+|warranty|guarantee|malfunction\w*|replace\w*|"
            r"repair\w*|expert (report|opinion)|not as (promised|described))\b", re.IGNORECASE),
    },
    "unfair_trade_practice": {
        "question": "Whether the opposite party adopted an unfair trade practice",
        "kind": "merits",
        "burden": "complainant",
        "sections": {"2(47)", "2(1)(r)"},
        "concepts": {"unfair_trade_practice"},
        "cues": re.compile(
            r"\b(unfair trade practice|misrepresent\w+|misleading|false (claim|advert\w+|representation)|"
            r"brochure|overcharg\w+|concealed|suppress\w+)\b", re.IGNORECASE),
    },
    "relief": {
        "question": "What relief, if any, the complainant is entitled to",
        "kind": "relief",
        "burden": "complainant",
        "sections": {"39", "14", "71", "72"},
        "concepts": set(),
        "cues": re.compile(
            r"\b(refund|compensation|interest|damages|mental agony|harass\w+|litigation cost|"
            r"costs?|replace\w*|possession be (delivered|handed)|entitled to)\b", re.IGNORECASE),
    },
}

# Threshold issues are ordered before merits, merits before relief, because a
# tribunal that answers a threshold issue against the complainant never reaches
# the rest.
KIND_ORDER = {"threshold": 0, "merits": 1, "relief": 2}

# --------------------------------------------------------------------------
# Strength scoring. Exposed for the ablation table.
# --------------------------------------------------------------------------
STRENGTH_WEIGHTS = {
    "grounding": 0.30,      # contentions actually pleaded, traceable to the case file
    "statutory": 0.30,      # verified provisions whose elements the facts satisfy
    "precedential": 0.25,   # admissible precedents whose outcome favours this side
    "unrebutted": 0.15,     # contentions the other side never answered
}

# Tribunal voice. Agent 1 keeps the court's `legal_reasoning` out of the fact
# list, but some of the court's own findings land in `procedural` ("As far as
# pecuniary jurisdiction is concerned... this Commission has the requisite
# pecuniary jurisdiction"). Attributing those to a party is wrong twice over:
# the court is not a party, and on a decided judgment the court's threshold
# findings ARE part of the outcome this pipeline is meant to predict. They are
# therefore excluded from the argument map and reported separately.
TRIBUNAL_VOICE = re.compile(
    r"\b(this (commission|forum|court|bench)\s+"
    r"(has|have|had|is|are|was|does|do|did|finds?|holds?|would|will|shall|can|cannot|lacks?)\b|"
    r"we (are|have|find|hold|do not|therefore)\b|"
    r"in our (considered )?(view|opinion)|we are of the (considered )?(view|opinion)|"
    r"as far as .{0,60} is concerned|it (is|would) (thus |therefore )?(be seen|follow|held|clear)|"
    r"therefore,? this|is (hereby )?(allowed|dismissed)|the complaint is (partly )?(allowed|dismissed))\b",
    re.IGNORECASE)

# Reported speech beats tribunal voice. "The preliminary objection taken by the
# OP is that this Commission does not have pecuniary jurisdiction" mentions the
# Commission but is the OP's plea being recorded, not the court's finding.
# Without this guard the defence disappears from the map entirely.
REPORTED_SPEECH = re.compile(
    r"\b(objections?\s+(taken|raised)|(taken|raised|filed)\s+by\s+the|contend(s|ed|ing)?\s+that|"
    r"submitt?ed\s+that|plead(s|ed)?\s+that|argued\s+that|according\s+to\s+the|"
    r"it\s+is\s+(contended|submitted|pleaded|urged)|(has|have)\s+been\s+resisted|"
    r"case\s+of\s+the\s+(complainants?|opposite\s+part(y|ies)))\b", re.IGNORECASE)

# A party-voiced procedural plea reads like an objection. Without either a
# party cue or an objection cue, a procedural sentence is left unattributed
# rather than being handed to the opposite party by default.
OBJECTION_CUE = re.compile(
    r"\b(preliminary objection|objection|barred by|not maintainable|maintainab\w+|"
    r"contend\w*|submit\w*|plead\w*|resisted|denied)\b", re.IGNORECASE)

# A contention is treated as "rebutted" when the other side has any contention
# on the same issue. This is issue-level, not sentence-level: rule-based
# sentence-to-sentence rebuttal pairing on legal prose produces more noise than
# signal, and a claim that the other side answered on the same issue is the
# honest unit of analysis here.
GROUNDING_SATURATION = 3      # contentions beyond this add nothing
LEANING_MARGIN = 0.10         # below this the issue is reported as contested


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------
@dataclass
class Contention:
    """One contention, traceable to a sentence Agent 1 extracted."""
    fact_id: str
    text: str
    party: str                      # complainant | opposite_party
    kind: str = "pleaded"           # pleaded | factual_assertion
    paragraph: int | None = None
    span: tuple | None = None
    label: str = ""                 # Agent 1's label (claim / defence / procedural / deficiency ...)
    rebutted: bool = False
    rebutted_by: list = field(default_factory=list)


@dataclass
class StatutorySupport:
    citation: str
    section: str
    role: str
    element_coverage: float
    unsatisfied_elements: list = field(default_factory=list)
    verified: bool = False
    counts_toward_strength: bool = False
    reason: str = ""


@dataclass
class PrecedentSupport:
    case_number: str | None
    forum: str | None
    outcome: str
    favours: str
    holding: str = ""
    authority_weight: float = 0.0
    shared_signals: list = field(default_factory=list)
    distinguishing_factors: list = field(default_factory=list)
    verified: bool = False
    counts_toward_strength: bool = False
    reason: str = ""


@dataclass
class SideCase:
    party: str
    contentions: list = field(default_factory=list)          # list[Contention]
    statutory_support: list = field(default_factory=list)    # list[StatutorySupport]
    precedent_support: list = field(default_factory=list)    # list[PrecedentSupport]
    strength: float = 0.0
    strength_components: dict = field(default_factory=dict)
    unrebutted_contentions: list = field(default_factory=list)
    gaps: list = field(default_factory=list)


@dataclass
class Issue:
    id: str
    question: str
    kind: str                      # threshold | merits | relief
    burden_on: str
    raised_by: list = field(default_factory=list)   # what put this issue in play
    complainant: SideCase = field(default_factory=lambda: SideCase("complainant"))
    opposite_party: SideCase = field(default_factory=lambda: SideCase("opposite_party"))
    leaning: str = "contested"     # complainant | opposite_party | contested | unsupported
    margin: float = 0.0
    confidence: float = 0.0
    dispositive: bool = False      # threshold issue leaning against the complainant
    one_sided_record: bool = False  # a side holds admissible support but pleaded nothing
    notes: list = field(default_factory=list)


@dataclass
class ArgumentMap:
    issues: list = field(default_factory=list)              # list[Issue]
    argument_balance: float = 0.0       # -1 (opposite party) .. +1 (complainant)
    merits_balance: float = 0.0
    threshold_risk: float = 0.0         # 0 (no threshold risk) .. 1 (likely dispositive)
    dispositive_risks: list = field(default_factory=list)
    unrebutted_by_party: dict = field(default_factory=dict)
    evidential_gaps: list = field(default_factory=list)
    excluded_support: list = field(default_factory=list)     # unverified, shown but not counted
    excluded_tribunal_findings: list = field(default_factory=list)  # court's voice, not a party's
    unattributed_contentions: list = field(default_factory=list)
    inputs_used: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def for_prediction(self) -> dict:
        """The feature set the Prediction Agent conditions on."""
        return {
            "argument_balance": self.argument_balance,
            "merits_balance": self.merits_balance,
            "threshold_risk": self.threshold_risk,
            "dispositive_risks": [r["issue_id"] for r in self.dispositive_risks],
            "issues": [
                {
                    "id": i.id, "kind": i.kind, "leaning": i.leaning,
                    "margin": round(i.margin, 3), "confidence": round(i.confidence, 3),
                    "complainant_strength": round(i.complainant.strength, 3),
                    "opposite_party_strength": round(i.opposite_party.strength, 3),
                }
                for i in self.issues
            ],
            "evidential_gap_count": len(self.evidential_gaps),
        }

    def summary(self) -> str:
        L = [f"Inputs used: {', '.join(self.inputs_used) or 'none'}",
             f"Argument balance: {self.argument_balance:+.3f} "
             f"(merits {self.merits_balance:+.3f}, threshold risk {self.threshold_risk:.3f})"]
        if self.dispositive_risks:
            L.append("DISPOSITIVE RISK - a threshold issue leans against the complainant:")
            for r in self.dispositive_risks:
                L.append(f"  - {r['issue_id']}: {r['note']}")
        L.append(f"\nISSUES ({len(self.issues)})")
        for n, i in enumerate(self.issues, 1):
            L.append(f"\n{n}. [{i.kind}] {i.question}")
            L.append(f"   raised by: {', '.join(i.raised_by) or '-'}   |   burden: {i.burden_on}")
            caveat = "  [one-sided pleading record]" if i.one_sided_record else ""
            L.append(f"   leaning: {i.leaning} (margin {i.margin:+.3f}, "
                     f"confidence {i.confidence:.2f}){caveat}")
            for side in (i.complainant, i.opposite_party):
                L.append(f"   {side.party.upper()}  strength {side.strength:.3f}  "
                         + "  ".join(f"{k}={v:.2f}" for k, v in side.strength_components.items()))
                for c in side.contentions:
                    mark = "" if c.rebutted else "  [UNREBUTTED]"
                    tag = "" if c.kind == "pleaded" else " (fact)"
                    L.append(f"     ({c.fact_id}{tag} ¶{c.paragraph}) {c.text[:130]}{mark}")
                for s in side.statutory_support:
                    flag = "" if s.counts_toward_strength else "  [not counted: " + s.reason + "]"
                    L.append(f"     statute: {s.citation} (elements {s.element_coverage:.0%}){flag}")
                for pr in side.precedent_support:
                    flag = "" if pr.counts_toward_strength else "  [not counted: " + pr.reason + "]"
                    L.append(f"     precedent: {pr.case_number} ({pr.forum}) {pr.outcome}{flag}")
                    if pr.holding:
                        L.append(f"        holding: {pr.holding[:140]}")
                for g in side.gaps:
                    L.append(f"     gap: {g}")
            for note in i.notes:
                L.append(f"   note: {note}")
        if self.evidential_gaps:
            L.append(f"\nEVIDENTIAL GAPS ({len(self.evidential_gaps)})")
            for g in self.evidential_gaps:
                L.append(f"  - {g}")
        if self.excluded_tribunal_findings:
            L.append(f"\nEXCLUDED - TRIBUNAL'S OWN VOICE ({len(self.excluded_tribunal_findings)})")
            for e in self.excluded_tribunal_findings:
                L.append(f"  - {e}")
        if self.unattributed_contentions:
            L.append(f"\nEXCLUDED - UNATTRIBUTABLE ({len(self.unattributed_contentions)})")
            for e in self.unattributed_contentions:
                L.append(f"  - {e}")
        if self.excluded_support:
            L.append("\nSUPPORT SHOWN BUT NOT COUNTED (failed verification)")
            for e in self.excluded_support:
                L.append(f"  - {e}")
        L.append("\nFOR PREDICTION AGENT")
        L.append(f"  {self.for_prediction()}")
        for w in self.warnings:
            L.append(f"WARNING: {w}")
        return "\n".join(L)


# --------------------------------------------------------------------------
# Helper functions (pure, individually testable)
# --------------------------------------------------------------------------
def classify_contention(text: str) -> list[str]:
    """Which issues a pleaded sentence speaks to (a sentence may touch several)."""
    return [iid for iid, meta in ISSUE_TEMPLATES.items() if meta["cues"].search(text)]


def party_for_label(label: str) -> str:
    """Agent 1's sentence label -> the party whose contention it is."""
    return "opposite_party" if label == "defence" else "complainant"


def grounding_score(n_contentions: int) -> float:
    return min(1.0, n_contentions / GROUNDING_SATURATION) if n_contentions else 0.0


def leaning_from(complainant: float, opposite: float) -> tuple[str, float, float]:
    """Direction, margin and confidence of an issue from the two side scores."""
    margin = complainant - opposite
    if complainant == 0.0 and opposite == 0.0:
        return "unsupported", 0.0, 0.0
    if abs(margin) < LEANING_MARGIN:
        return "contested", margin, round(min(complainant, opposite), 3)
    direction = "complainant" if margin > 0 else "opposite_party"
    return direction, margin, round(min(1.0, abs(margin) + max(complainant, opposite) / 2), 3)


# --------------------------------------------------------------------------
# The agent
# --------------------------------------------------------------------------
LLM_SYSTEM = (
    "You are a legal argument-mining assistant for Indian consumer disputes. You are given the "
    "issues framed in a pending case and a numbered list of contentions already extracted "
    "verbatim from the case file. Assign each contention to the issues it actually speaks to. "
    'Respond ONLY with JSON: [{"id": <int>, "issues": ["<issue_id>", ...]}]. '
    "Use only the issue ids given. Never add, reword or invent a contention."
)


class ArgumentAnalysisAgent(BaseAgent):
    """
    Agent 4: turns facts, statutes and precedents into a structured argument map.

    Usage inside the pipeline (LangGraph-compatible, same contract as Agents 1-3):
        state = FactExtractionAgent().run(state)
        state = StatuteRetrievalAgent(conn=conn).run(state)
        state = PrecedentRetrievalAgent(conn=conn).run(state)
        state = ArgumentAnalysisAgent().run(state)     # -> state["argument_map"]
    """
    name = "argument_analysis_agent"

    def __init__(self, llm=None, max_precedents_per_issue: int = 3,
                 count_unverified: bool = False):
        """
        count_unverified: let unverified provisions and inadmissible precedents
            contribute to the strength scores. Default False - an argument
            should not be scored on authority the system could not
            authenticate. Available for ablation, not for production use.
        """
        self.llm = llm
        self.max_precedents_per_issue = max_precedents_per_issue
        self.count_unverified = count_unverified
        self.llm_error: str | None = None

    # ---- pipeline interface -------------------------------------------------
    def run(self, state: dict) -> dict:
        amap = self.analyse(
            fact_sheet=state.get("fact_sheet"),
            statute_bundle=state.get("statute_bundle"),
            precedent_bundle=state.get("precedent_bundle"),
        )
        state["argument_map"] = amap
        state["argument_features"] = amap.for_prediction()
        return state

    # ---- main entry ---------------------------------------------------------
    def analyse(self, fact_sheet=None, statute_bundle=None, precedent_bundle=None) -> ArgumentMap:
        amap = ArgumentMap()
        amap.inputs_used = [n for n, v in (("fact_sheet", fact_sheet),
                                           ("statute_bundle", statute_bundle),
                                           ("precedent_bundle", precedent_bundle)) if v is not None]
        if fact_sheet is None:
            amap.warnings.append("No fact sheet supplied - run the Fact Extraction Agent first.")
            return amap
        if statute_bundle is None:
            amap.warnings.append(
                "No statute bundle - issues framed from the pleadings alone and no statutory "
                "support attached; run the Statute Retrieval Agent for a complete map.")
        if precedent_bundle is None:
            amap.warnings.append(
                "No precedent bundle - no precedential support attached; run the Precedent "
                "Retrieval Agent for a complete map.")

        # --- 1. frame the issues actually in play
        issues = self._frame_issues(fact_sheet, statute_bundle)
        if not issues:
            amap.warnings.append("Nothing in the case raised a recognised issue.")
            return amap

        # --- 2. route the pleaded contentions to those issues
        contentions = self._collect_contentions(fact_sheet, amap)
        assignment = self._assign(contentions, issues, amap)
        for issue in issues:
            for c, iids in zip(contentions, assignment):
                if issue.id not in iids:
                    continue
                side = issue.complainant if c.party == "complainant" else issue.opposite_party
                side.contentions.append(c)

        # --- 3. attach support, detect rebuttal, score
        for issue in issues:
            self._attach_statutes(issue, statute_bundle, amap)
            self._attach_precedents(issue, precedent_bundle, amap)
            self._mark_rebuttal(issue)
            self._score(issue, statute_bundle)

        issues.sort(key=lambda i: (KIND_ORDER.get(i.kind, 3), -abs(i.margin)))
        amap.issues = issues

        # --- 4. aggregate
        self._aggregate(amap, statute_bundle)
        return amap

    # ---- internals ----------------------------------------------------------
    @staticmethod
    def _frame_issues(fact_sheet, statute_bundle) -> list[Issue]:
        """
        An issue is raised when something in this case raises it: a provision
        Agent 2 returned, a concept hook Agent 1 emitted, or a pleaded sentence
        that speaks to it. Issues nothing raises are not framed - the map is of
        this dispute, not of the Act.
        """
        raised: dict[str, list[str]] = {}

        for m in (statute_bundle.matches if statute_bundle else []):
            for iid, meta in ISSUE_TEMPLATES.items():
                if m.section in meta["sections"]:
                    raised.setdefault(iid, []).append(f"statute {m.citation}")

        for h in (fact_sheet.legal_hooks or []):
            for iid, meta in ISSUE_TEMPLATES.items():
                if h["concept"] in meta["concepts"]:
                    raised.setdefault(iid, []).append(f"hook {h['concept']}")

        pleaded = fact_sheet.claims + fact_sheet.defences + fact_sheet.procedural_points
        for f in pleaded:
            for iid in classify_contention(f.text):
                raised.setdefault(iid, []).append(f"pleaded in {f.id}")

        # Relief is always live once anything is claimed - a complaint that
        # establishes liability still needs the quantum question answered.
        if fact_sheet.claims and "relief" not in raised:
            raised["relief"] = ["claims pleaded"]

        issues = []
        for iid, sources in raised.items():
            meta = ISSUE_TEMPLATES[iid]
            issues.append(Issue(
                id=iid, question=meta["question"], kind=meta["kind"], burden_on=meta["burden"],
                raised_by=sorted(dict.fromkeys(sources))[:5],
                complainant=SideCase("complainant"), opposite_party=SideCase("opposite_party"),
            ))
        return issues

    @staticmethod
    def _collect_contentions(fact_sheet, amap=None) -> list[Contention]:
        """
        Three sources, because a party's case is not only what it prays for.

        - `claims` / `defences`: what each side contends (kind="pleaded").
        - `facts`: the complainant's factual averments (kind="factual_assertion").
          A merits issue is carried by these, not by the prayer - "the seller
          failed to replace the defective product" IS the case on deficiency,
          while "I seek a refund" is only the relief sought. Scoring merits
          issues off pleaded sentences alone leaves an uncontested complaint
          with a zero merits balance, which is exactly backwards.
        - `procedural_points`: limitation and jurisdiction pleas, attributed to
          the party the wording indicates and otherwise to the opposite party,
          which is who normally raises them.

        Attribution caveat: when the input is a judgment, the narrative is the
        tribunal's summary of the complainant's case, so treating factual
        averments as the complainant's is right for that input. For a
        two-sided pleading bundle, facts drawn from the written version would
        need attributing to the opposite party - the `party` field is there
        for that, and is not inferred from the sentence.
        """
        out, tribunal, unattributed = [], [], []

        def is_tribunal(text: str) -> bool:
            return bool(TRIBUNAL_VOICE.search(text)) and not REPORTED_SPEECH.search(text)

        def keep(f, party, kind):
            if is_tribunal(f.text):
                tribunal.append(f"{f.id}: {f.text[:120]}")
                return
            out.append(Contention(fact_id=f.id, text=f.text, party=party, kind=kind,
                                  paragraph=f.paragraph, span=f.span, label=f.label))

        for f in fact_sheet.claims:
            keep(f, "complainant", "pleaded")
        for f in fact_sheet.defences:
            keep(f, "opposite_party", "pleaded")
        for f in fact_sheet.facts:
            keep(f, "complainant", "factual_assertion")
        for f in fact_sheet.procedural_points:
            if is_tribunal(f.text):
                tribunal.append(f"{f.id}: {f.text[:120]}")
                continue
            op_side = re.search(
                r"\b(o\.?p\.?|opposite part(y|ies)|developer|builder|respondent|insurer|bank)\b",
                f.text, re.IGNORECASE)
            complainant_side = re.search(
                r"\b(complainants?\b|recurrent cause of action|continuing cause of action|condon\w+)\b",
                f.text, re.IGNORECASE)
            if op_side and not complainant_side:
                party = "opposite_party"
            elif complainant_side and not op_side:
                party = "complainant"
            elif OBJECTION_CUE.search(f.text):
                party = "opposite_party"   # an objection is the OP's to take
            else:
                unattributed.append(f"{f.id}: {f.text[:120]}")
                continue
            out.append(Contention(fact_id=f.id, text=f.text, party=party,
                                  kind="pleaded", paragraph=f.paragraph, span=f.span, label=f.label))

        if amap is not None:
            amap.excluded_tribunal_findings = tribunal
            amap.unattributed_contentions = unattributed
            if tribunal:
                amap.warnings.append(
                    f"{len(tribunal)} sentence(s) in the tribunal's own voice were excluded from the "
                    "argument map - the court is not a party, and on a decided judgment its findings "
                    "are part of the outcome being predicted.")
            if unattributed:
                amap.warnings.append(
                    f"{len(unattributed)} procedural sentence(s) could not be attributed to a party "
                    "and were left out rather than assigned by default.")
        return out

    def _assign(self, contentions: list[Contention], issues: list[Issue],
                amap: ArgumentMap) -> list[list[str]]:
        """Rule-based issue assignment, optionally re-done by an LLM."""
        live = {i.id for i in issues}
        base = [[iid for iid in classify_contention(c.text) if iid in live] for c in contentions]
        if self.llm is None:
            return base
        try:
            issue_list = "\n".join(f"{i.id}: {i.question}" for i in issues)
            listing = "\n".join(f"{n}: [{c.party}] {c.text}" for n, c in enumerate(contentions))
            reply = self.llm.complete_json(
                LLM_SYSTEM, f"ISSUES\n{issue_list}\n\nCONTENTIONS\n{listing}")
            for item in reply:
                n = int(item["id"])
                if not 0 <= n < len(base):
                    continue
                chosen = [iid for iid in item.get("issues", []) if iid in live]
                if chosen:
                    base[n] = chosen
        except Exception as e:  # noqa: BLE001 - any failure => rule assignment
            self.llm_error = str(e)
            amap.warnings.append(f"LLM argument mapping unavailable ({e}); using rule-based mapping.")
        return base

    def _attach_statutes(self, issue: Issue, statute_bundle, amap: ArgumentMap):
        """
        Attach provisions to the side they help.

        A provision's `favours` is the direction it cuts: limitation helps the
        opposite party even though it is the complainant who must show the
        complaint is in time. Neutral provisions (definitions, jurisdiction)
        are attached to the side bearing the burden on that issue.
        """
        meta = ISSUE_TEMPLATES[issue.id]
        for m in (statute_bundle.matches if statute_bundle else []):
            if m.section not in meta["sections"]:
                continue
            counts = bool(m.verified and m.admissible_as_authority)
            reason = ""
            if not counts:
                reason = ("not verified by CiteVerify" if not m.verified
                          else "section numbering not verified against the gazette")
                amap.excluded_support.append(f"{m.citation} on issue '{issue.id}' - {reason}")
            support = StatutorySupport(
                citation=m.citation, section=m.section, role=m.role,
                element_coverage=m.element_coverage,
                unsatisfied_elements=list(m.unsatisfied_elements),
                verified=m.verified,
                counts_toward_strength=counts or self.count_unverified,
                reason=reason,
            )
            target = (issue.opposite_party if m.favours == "opposite_party"
                      else issue.complainant if m.favours == "complainant"
                      else (issue.complainant if issue.burden_on == "complainant"
                            else issue.opposite_party))
            target.statutory_support.append(support)

    def _attach_precedents(self, issue: Issue, precedent_bundle, amap: ArgumentMap):
        """
        Attach precedents to the side their operative outcome favours.

        A dismissed complaint on comparable facts is authority for the opposite
        party; an allowed one is authority for the complainant. Merits issues
        take precedents generally; threshold issues take only precedents whose
        holdings actually address that threshold, so a builder-delay authority
        is not silently treated as authority on limitation.
        """
        if precedent_bundle is None:
            return
        meta = ISSUE_TEMPLATES[issue.id]
        attached = 0
        for m in precedent_bundle.matches:
            holding = next((h.text for h in m.holdings if meta["cues"].search(h.text)), "")
            if issue.kind == "threshold" and not holding:
                continue  # no holding on this threshold -> not authority on it
            if not holding and not (m.shared_signals or issue.kind == "relief"):
                continue
            if attached >= self.max_precedents_per_issue:
                break
            counts = bool(m.verified and m.admissible_as_authority)
            reason = ""
            if not counts:
                reason = ("not verified against the curated corpus" if not m.verified
                          else "verified but no holding located - not usable as authority")
                amap.excluded_support.append(
                    f"precedent {m.case_number} on issue '{issue.id}' - {reason}")
            support = PrecedentSupport(
                case_number=m.case_number, forum=m.forum, outcome=m.outcome.label,
                favours=m.outcome.favours, holding=holding or (m.holdings[0].text if m.holdings else ""),
                authority_weight=authority_weight(m.forum),
                shared_signals=list(m.shared_signals),
                distinguishing_factors=list(m.distinguishing_factors),
                verified=m.verified,
                counts_toward_strength=counts or self.count_unverified,
                reason=reason,
            )
            if m.outcome.favours == "opposite_party":
                issue.opposite_party.precedent_support.append(support)
            elif m.outcome.favours == "complainant":
                issue.complainant.precedent_support.append(support)
            else:
                continue
            attached += 1

    @staticmethod
    def _mark_rebuttal(issue: Issue):
        """
        Issue-level rebuttal: a contention is met if the other side contends
        anything on the same issue. What is left unanswered is recorded, because
        an uncontroverted averment is a real evidentiary point, not a rhetorical
        one.
        """
        c_ids = [c.fact_id for c in issue.complainant.contentions]
        o_ids = [c.fact_id for c in issue.opposite_party.contentions]
        for c in issue.complainant.contentions:
            c.rebutted = bool(o_ids)
            c.rebutted_by = list(o_ids[:3])
        for c in issue.opposite_party.contentions:
            c.rebutted = bool(c_ids)
            c.rebutted_by = list(c_ids[:3])
        issue.complainant.unrebutted_contentions = [c.fact_id for c in issue.complainant.contentions
                                                    if not c.rebutted]
        issue.opposite_party.unrebutted_contentions = [c.fact_id for c in issue.opposite_party.contentions
                                                       if not c.rebutted]

    def _score(self, issue: Issue, statute_bundle):
        """Four transparent components per side; nothing unscored, nothing hidden."""
        for side in (issue.complainant, issue.opposite_party):
            statutes = [s for s in side.statutory_support if s.counts_toward_strength]
            precedents = [p for p in side.precedent_support if p.counts_toward_strength]
            comp = {
                "grounding": grounding_score(len(side.contentions)),
                "statutory": (sum(s.element_coverage for s in statutes) / len(statutes)
                              if statutes else 0.0),
                "precedential": (sum(p.authority_weight for p in precedents) / len(precedents)
                                 if precedents else 0.0),
                "unrebutted": (len(side.unrebutted_contentions) / len(side.contentions)
                               if side.contentions else 0.0),
            }
            # A side that pleaded nothing on the issue scores nothing, whatever
            # support happens to sit on the file: an argument has to be made.
            side.strength = round(sum(STRENGTH_WEIGHTS[k] * v for k, v in comp.items()), 4) \
                if side.contentions else 0.0
            side.strength_components = {k: round(v, 3) for k, v in comp.items()}
            side.gaps = self._side_gaps(side)

        issue.leaning, issue.margin, issue.confidence = leaning_from(
            issue.complainant.strength, issue.opposite_party.strength)

        # One-sided pleading record. A side can hold an on-point, admissible
        # authority and still score zero because it pleaded nothing on the
        # issue. That is usually an artefact of the input: in a decided
        # judgment the answering argument is voiced by the tribunal and is
        # excluded above. Prediction should discount such a leaning, so it is
        # flagged rather than presented as a settled imbalance.
        for side, other in ((issue.complainant, issue.opposite_party),
                            (issue.opposite_party, issue.complainant)):
            if not side.contentions and other.contentions and (
                    [s for s in side.statutory_support if s.counts_toward_strength]
                    or [p for p in side.precedent_support if p.counts_toward_strength]):
                issue.one_sided_record = True
                issue.notes.append(
                    f"one-sided pleading record: the {side.party} has admissible support on this "
                    "issue but pleaded nothing on it, so it scores zero - common when the input is "
                    "a decided judgment, where the answering argument sits in the tribunal's voice")

        # A threshold issue that the complainant must establish and has not
        # can dispose of the case regardless of the merits.
        if issue.kind == "threshold" and issue.leaning == "opposite_party":
            issue.dispositive = True
            issue.notes.append(
                "threshold issue leaning against the complainant - capable of disposing of the "
                "complaint without reaching the merits")

        # Statutory findings Agent 2 computed are attached as notes where they
        # bear on the issue, so the reader sees the computed position too.
        if statute_bundle is not None:
            if issue.id == "limitation" and statute_bundle.limitation.status != "indeterminate":
                issue.notes.append(f"Agent 2 limitation finding: {statute_bundle.limitation.status} "
                                   f"- {statute_bundle.limitation.note}")
            if issue.id == "jurisdiction" and statute_bundle.jurisdiction.determinate:
                issue.notes.append(
                    f"Agent 2 jurisdiction finding: Rs.{statute_bundle.jurisdiction.amount_in_issue_inr:,} "
                    f"-> {statute_bundle.jurisdiction.forum}")

    @staticmethod
    def _side_gaps(side: SideCase) -> list[str]:
        gaps = []
        if side.contentions and not side.statutory_support:
            gaps.append("contentions pleaded but no statutory provision attached")
        if side.contentions and not side.precedent_support:
            gaps.append("no precedent supporting this side on this issue")
        for s in side.statutory_support:
            for el in s.unsatisfied_elements:
                gaps.append(f"{s.citation}: element not established - {el}")
        return gaps[:6]

    @staticmethod
    def _aggregate(amap: ArgumentMap, statute_bundle):
        """
        Two numbers, kept apart on purpose.

        `merits_balance` is who is winning on liability. `threshold_risk` is the
        chance the complaint never gets there. Averaging them would let a strong
        merits case mask a fatal limitation problem, which is precisely the
        error a prediction system must not make.
        """
        merits = [i for i in amap.issues if i.kind == "merits"]
        thresholds = [i for i in amap.issues if i.kind == "threshold"]

        if merits:
            amap.merits_balance = round(
                sum(i.complainant.strength - i.opposite_party.strength for i in merits) / len(merits), 4)
        if thresholds:
            risks = [i.opposite_party.strength - i.complainant.strength for i in thresholds]
            amap.threshold_risk = round(max(0.0, max(risks)), 4)

        amap.dispositive_risks = [
            {"issue_id": i.id, "question": i.question,
             "margin": round(i.margin, 3),
             "one_sided_record": i.one_sided_record,
             "note": i.notes[0] if i.notes else "threshold issue leaning to the opposite party"}
            for i in amap.issues if i.dispositive
        ]
        # The overall balance is the merits position discounted by threshold risk.
        amap.argument_balance = round(amap.merits_balance * (1.0 - amap.threshold_risk), 4)

        amap.unrebutted_by_party = {
            "complainant": sorted({fid for i in amap.issues for fid in i.complainant.unrebutted_contentions}),
            "opposite_party": sorted({fid for i in amap.issues for fid in i.opposite_party.unrebutted_contentions}),
        }
        amap.evidential_gaps = sorted({f"[{i.id}] {g}" for i in amap.issues
                                       for side in (i.complainant, i.opposite_party)
                                       for g in side.gaps})
        amap.excluded_support = sorted(dict.fromkeys(amap.excluded_support))

        if not merits:
            amap.warnings.append("No merits issue was framed - the balance reflects threshold issues only.")
