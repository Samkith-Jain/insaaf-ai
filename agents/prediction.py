"""
Agent 5 of the Multi-Agent Legal Reasoning Pipeline: PREDICTION AGENT.

Objective (project synopsis): predict the legal outcome with improved accuracy
and transparency - take the structured argument map, the statutory findings and
the precedential prior, and return a disposition, a calibrated confidence, the
issue-by-issue findings that produce it, and the reasons a reader can check.

Position in the pipeline
------------------------
    Fact Extraction -> Statute Retrieval -> Precedent Retrieval ->
    Argument Analysis -> [PREDICTION] -> Verification -> Explanation

It consumes `argument_map` (Agent 4), `precedent_bundle` (Agent 3),
`statute_bundle` (Agent 2) and `fact_sheet` (Agent 1), and writes `prediction`
into the shared state. Only the argument map is required; with fewer inputs the
prediction is thinner, less confident, and says so - it never fails silently.

How the prediction is made
--------------------------
The model is a transparent Bayesian update, not a black box, because a
prediction a tribunal user cannot interrogate is not usable:

    PRIOR      the authority-weighted rate at which comparable, VERIFIED
               precedents went for the complainant (Agent 3), shrunk toward a
               neutral prior so a corpus of two cases cannot produce certainty.

    UPDATE     the prior's log-odds are moved by this case's own evidence -
               the merits balance from the argument map, the statutory element
               coverage from Agent 2, and the share of contentions the other
               side never answered. Every term's contribution is reported in
               `drivers`, in log-odds, so the confidence score decomposes.

    GATE       threshold issues are decided FIRST, as a tribunal decides them.
               A complaint that is time-barred or filed in the wrong forum is
               dismissed without the merits being reached, so a fatal threshold
               issue short-circuits the merits computation rather than being
               averaged into it. `decided_at` records which stage disposed of
               the case.

    SPLIT      a favourable outcome is then split between `allowed` and
               `partly_allowed` - in consumer adjudication relief as prayed is
               less common than relief reduced, so predicting "allowed" for
               every complainant win would be wrong most of the time. The split
               is informed by the corpus distribution and by the strength of
               the relief issue.

    CALIBRATE  confidence is capped by the quality of the evidence behind it.
               A prediction resting on one unverified precedent and two pleaded
               sentences cannot be 95% confident, however lopsided the
               arithmetic. Below `abstain_threshold` the agent returns
               `indeterminate` and says what is missing instead of guessing.

Verification-first (Objective 4)
--------------------------------
Only precedents that passed CiteVerify contribute to the prior, and only
support Agent 4 counted contributes to the merits balance. Unverified authority
is listed in `excluded_from_prior` with a reason - visible, never silently
dropped and never silently relied on. A prediction is therefore never built on
authority the system could not authenticate.

Leakage
-------
The agent reads only the four upstream bundles. Agent 1 has already stripped the
operative order, Agent 3 has already excluded the case under consideration from
its own results, and Agent 4 has already excluded the tribunal's own voice. This
agent never touches `state["case_text"]`, so it cannot read the result it is
predicting. If Agent 1 reports that the outcome was NOT removed, the prediction
is still produced but flagged as leakage-suspect, because a backtest on such a
case measures nothing.

Rules vs. LLM
-------------
Offline and rule-based by default (same trade-off as Agents 1-4). Pass
`llm=AnthropicBackend()` to have an LLM assess the merits direction; its opinion
is admitted only as a BOUNDED log-odds adjustment (`LLM_MAX_LOGIT_SHIFT`) on top
of the grounded computation, and it is reported as its own driver. It can
therefore shade a borderline prediction but can never overturn the evidence or
manufacture confidence. Any failure falls back to the rules.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field, asdict

from agents.base import BaseAgent
from agents.precedent_retrieval import COMPLAINANT_FAVOURABLE, authority_weight

# --------------------------------------------------------------------------
# Outcome vocabulary. Deliberately the same labels Agent 3 reads off decided
# judgments (`OUTCOME_PATTERNS`), so a prediction can be compared directly with
# a real disposition in a backtest without a mapping layer.
# --------------------------------------------------------------------------
ALLOWED = "allowed"
PARTLY_ALLOWED = "partly_allowed"
DISMISSED = "dismissed"
INDETERMINATE = "indeterminate"

# --------------------------------------------------------------------------
# Prior.
#
# NEUTRAL_PRIOR is not an empirical claim about consumer litigation - it is a
# deliberately uninformative starting point, used only where the corpus cannot
# supply a rate. Fit it on the ingested corpus before reporting accuracy
# figures; leaving it at 0.5 means "the system has no opinion before it looks
# at the precedents", which is the honest default.
#
# PRIOR_PSEUDO_COUNT shrinks the observed rate toward that neutral point. Two
# precedents that both went for the complainant give a raw rate of 1.0; with a
# pseudo-count of 2 that becomes 0.75, which is what two cases actually
# support. This is the single most important guard against overconfidence on a
# small corpus, which is the regime this system operates in.
# --------------------------------------------------------------------------
NEUTRAL_PRIOR = 0.50
PRIOR_PSEUDO_COUNT = 2.0

# --------------------------------------------------------------------------
# Log-odds weights for the merits update. Exposed for the ablation table.
#
# Read them as: a fully one-sided merits balance (+1.0) multiplies the odds of
# a complainant win by e^2.2 ~= 9. That is strong but not decisive, which is
# correct - a well-argued case still loses on an unsatisfied element.
# --------------------------------------------------------------------------
POSTERIOR_WEIGHTS = {
    "merits_balance": 2.20,        # argument map, -1 .. +1
    "element_coverage": 1.10,      # statutory elements established, centred at 0.5
    "unrebutted_advantage": 0.60,  # contentions the other side never answered
    "gap_penalty": -0.45,          # per distinct unsatisfied statutory element, capped
}
GAP_PENALTY_CAP = 3               # beyond three gaps the penalty stops growing
LLM_MAX_LOGIT_SHIFT = 0.80        # an LLM may shade, never overturn

# --------------------------------------------------------------------------
# Threshold gate.
#
# THRESHOLD_GAIN converts an issue margin into a probability of failure. A
# margin of 0.3 against the complainant becomes sigmoid(3.0*0.3) = 0.71.
#
# UNSUPPORTED_THRESHOLD_RISK: an issue that was framed but that neither side
# argued. The burden is formally on the complainant, but tribunals do not
# dismiss for want of a maintainability plea nobody took, so an unargued
# threshold is a small risk, not an even-money one.
#
# ONE_SIDED_DISCOUNT: Agent 4 flags issues where a side holds admissible
# authority but pleaded nothing, which on a decided judgment usually means the
# answering argument sat in the tribunal's voice and was excluded. Such a
# margin is an artefact of the input, so it is halved before it can dispose of
# a case.
# --------------------------------------------------------------------------
THRESHOLD_GAIN = 3.0
UNSUPPORTED_THRESHOLD_RISK = 0.15
ONE_SIDED_DISCOUNT = 0.50
DISPOSITIVE_GATE = 0.60           # above this, the case is predicted to end here
RESIDUAL_THRESHOLD_WEIGHT = 1.0   # threshold risk below the gate still discounts the merits

# Hard statutory findings from Agent 2 are stronger evidence than argument
# scores, because they are computed from dates and amounts rather than inferred
# from prose. They set a floor on the relevant threshold risk.
LIMITATION_BARRED_FLOOR = 0.70
LIMITATION_BARRED_FLOOR_IF_CONTINUING = 0.40   # a pleaded continuing cause of action
FORUM_MISMATCH_FLOOR = 0.55

# --------------------------------------------------------------------------
# Disposition split.
#
# PARTLY_ALLOWED_SHARE is the fallback share of complainant wins that are
# partial rather than full, used when the corpus is too small to estimate it.
# It is a modelling assumption, stated here rather than buried: full relief as
# prayed (including the compensation figure claimed) is granted less often than
# relief in part, so a system that always predicted "allowed" would be wrong on
# most of the cases it got directionally right.
# --------------------------------------------------------------------------
PARTLY_ALLOWED_SHARE = 0.60
RELIEF_STRENGTH_PIVOT = 0.45      # a well-made relief case pulls toward full allowance

# --------------------------------------------------------------------------
# Calibration.
#
# Confidence is the directional certainty of the model, SCALED BY how much
# evidence stands behind it, then capped. No configuration of this agent can
# report certainty: a legal prediction system that says 100% is lying.
# --------------------------------------------------------------------------
CONFIDENCE_CEILING = 0.92
EVIDENCE_WEIGHTS = {
    "verified_precedents": 0.35,   # saturating at VERIFIED_PRECEDENT_SATURATION
    "grounded_contentions": 0.25,  # saturating at CONTENTION_SATURATION
    "statutory_grounding": 0.25,   # verified provisions with elements assessed
    "input_completeness": 0.15,    # how many of the four bundles were supplied
}
VERIFIED_PRECEDENT_SATURATION = 4
CONTENTION_SATURATION = 8
DEFAULT_ABSTAIN_THRESHOLD = 0.35   # below this confidence, return `indeterminate`


# --------------------------------------------------------------------------
# Small numerical helpers (pure, individually testable)
# --------------------------------------------------------------------------
def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-min(x, 60.0)))
    e = math.exp(max(x, -60.0))
    return e / (1.0 + e)


def logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def shrink(rate: float | None, n: int, neutral: float = NEUTRAL_PRIOR,
           pseudo: float = PRIOR_PSEUDO_COUNT) -> float:
    """
    Beta-style shrinkage of an observed rate toward a neutral prior.

    With no observations the result is `neutral`; with many it approaches the
    observed rate. This is what stops three retrieved precedents from being
    treated as a settled line of authority.
    """
    if rate is None or n <= 0:
        return neutral
    return (rate * n + neutral * pseudo) / (n + pseudo)


def median_or_none(values: list) -> float | None:
    vals = [v for v in values if isinstance(v, (int, float))]
    return round(float(statistics.median(vals)), 2) if vals else None


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------
@dataclass
class Driver:
    """
    One named contribution to the prediction, in log-odds.

    This is the transparency contract: the posterior is exactly the prior plus
    the sum of the drivers, so a reader can audit the confidence score term by
    term instead of being asked to trust it.
    """
    name: str
    value: float                    # the underlying feature value
    logit_contribution: float       # how far it moved the log-odds
    direction: str                  # complainant | opposite_party | neutral
    explanation: str = ""
    grounded_in: list = field(default_factory=list)   # fact ids / citations


@dataclass
class IssuePrediction:
    """The predicted finding on a single framed issue."""
    issue_id: str
    question: str
    kind: str                       # threshold | merits | relief
    predicted_for: str              # complainant | opposite_party | too_close_to_call
    probability: float              # P(decided for the complainant)
    margin: float
    dispositive: bool = False
    one_sided_record: bool = False
    basis: list = field(default_factory=list)
    caveats: list = field(default_factory=list)


@dataclass
class ReliefForecast:
    """
    An indicative forecast of what would be awarded IF the complaint succeeds.

    Every head is grounded twice over: it must have been claimed in the case
    file, and it must appear in the relief actually granted by the verified
    precedents. A head the complainant never sought is not forecast, and a head
    no comparable decision has ever granted is reported as unsupported rather
    than invented. Amounts are indicative of the corpus, not an award.
    """
    conditional_on: str = "complaint succeeding in whole or part"
    relief_heads: list = field(default_factory=list)
    claimed_heads_without_precedent: list = field(default_factory=list)
    principal_claimed_inr: int | None = None
    typical_interest_rate_pct: float | None = None
    interest_rate_basis: list = field(default_factory=list)
    precedent_amounts_inr: list = field(default_factory=list)
    note: str = ""


@dataclass
class Counterfactual:
    """What would have to change for the prediction to change, and by how much."""
    change: str
    revised_probability: float
    revised_disposition: str
    flips_prediction: bool


@dataclass
class Prediction:
    # --- the answer
    disposition: str = INDETERMINATE          # allowed | partly_allowed | dismissed | indeterminate
    favours: str = "unknown"                  # complainant | opposite_party | unknown
    probability_complainant_succeeds: float = 0.0
    confidence: float = 0.0
    decided_at: str = "merits"                # threshold | merits | abstained
    disposition_probabilities: dict = field(default_factory=dict)

    # --- how it was reached
    prior: float = NEUTRAL_PRIOR
    prior_basis: str = ""
    posterior_logit: float = 0.0
    drivers: list = field(default_factory=list)            # list[Driver]
    issue_predictions: list = field(default_factory=list)  # list[IssuePrediction]
    dispositive_grounds: list = field(default_factory=list)
    threshold_risk: float = 0.0

    # --- what follows from it
    relief_forecast: ReliefForecast = field(default_factory=ReliefForecast)
    counterfactuals: list = field(default_factory=list)    # list[Counterfactual]

    # --- how far it can be trusted
    evidence_quality: float = 0.0
    evidence_components: dict = field(default_factory=dict)
    abstained: bool = False
    abstention_reasons: list = field(default_factory=list)
    excluded_from_prior: list = field(default_factory=list)
    evidential_gaps: list = field(default_factory=list)
    leakage_suspect: bool = False
    inputs_used: list = field(default_factory=list)
    ranked_by: str = "rules"
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    # ---- downstream contracts ----------------------------------------------
    def for_verification(self) -> dict:
        """
        What the Verification Agent (CiteVerify / SPMA) re-checks before this
        prediction may be shown: every authority the prediction leaned on.
        """
        return {
            "disposition": self.disposition,
            "authorities_relied_on": [d.grounded_in for d in self.drivers if d.grounded_in],
            "excluded_from_prior": self.excluded_from_prior,
            "abstained": self.abstained,
        }

    def for_explanation(self) -> dict:
        """
        The IRAC-ready payload for the Explanation Agent.

        Issues become the I and A of IRAC, the drivers become the reasoning,
        the gaps become what the explanation must concede, and the
        counterfactuals become what the user can act on.
        """
        return {
            "disposition": self.disposition,
            "confidence": round(self.confidence, 3),
            "decided_at": self.decided_at,
            "issues": [
                {"issue_id": i.issue_id, "question": i.question, "kind": i.kind,
                 "predicted_for": i.predicted_for, "probability": round(i.probability, 3),
                 "basis": i.basis, "caveats": i.caveats}
                for i in self.issue_predictions
            ],
            "reasons": [
                {"name": d.name, "direction": d.direction,
                 "weight": round(d.logit_contribution, 3), "explanation": d.explanation}
                for d in self.drivers if abs(d.logit_contribution) > 1e-9
            ],
            "dispositive_grounds": self.dispositive_grounds,
            "must_concede": self.evidential_gaps,
            "relief": asdict(self.relief_forecast),
            "counterfactuals": [asdict(c) for c in self.counterfactuals],
        }

    def summary(self) -> str:
        L = [f"Inputs used: {', '.join(self.inputs_used) or 'none'}",
             ""]
        if self.abstained:
            L.append("PREDICTION: INDETERMINATE - the agent declined to predict.")
            for r in self.abstention_reasons:
                L.append(f"  - {r}")
        else:
            L.append(f"PREDICTED DISPOSITION: {self.disposition.upper().replace('_', ' ')} "
                     f"(favours the {self.favours.replace('_', ' ')})")
        L.append(f"  P(complainant succeeds) = {self.probability_complainant_succeeds:.3f}   "
                 f"confidence = {self.confidence:.3f}   decided at: {self.decided_at}")
        if self.disposition_probabilities:
            L.append("  disposition probabilities: " + "  ".join(
                f"{k}={v:.3f}" for k, v in self.disposition_probabilities.items()))
        if self.leakage_suspect:
            L.append("  ** LEAKAGE SUSPECT: the operative order was not stripped upstream - "
                     "do not use this case in a backtest **")

        if self.dispositive_grounds:
            L.append(f"\nDISPOSITIVE GROUND(S) - the case is predicted to end before the merits")
            for g in self.dispositive_grounds:
                L.append(f"  - [{g['issue_id']}] {g['question']}")
                L.append(f"      P(decided against the complainant) = {g['probability']:.3f}  "
                         f"({g['basis']})")

        L.append(f"\nHOW THIS WAS REACHED")
        L.append(f"  prior  {self.prior:.3f}  (logit {logit(self.prior):+.3f})  - {self.prior_basis}")
        for d in self.drivers:
            if abs(d.logit_contribution) < 1e-9:
                continue
            L.append(f"  {d.logit_contribution:+.3f}  {d.name} (value {d.value:+.3f}, "
                     f"favours {d.direction})")
            if d.explanation:
                L.append(f"          {d.explanation}")
        L.append(f"  = posterior logit {self.posterior_logit:+.3f} "
                 f"-> {sigmoid(self.posterior_logit):.3f} on the merits")
        if self.threshold_risk:
            L.append(f"  x threshold discount (risk {self.threshold_risk:.3f}) "
                     f"-> {self.probability_complainant_succeeds:.3f}")

        if self.issue_predictions:
            L.append(f"\nISSUE-BY-ISSUE FINDINGS ({len(self.issue_predictions)})")
            for i in self.issue_predictions:
                mark = "  [DISPOSITIVE]" if i.dispositive else ""
                L.append(f"  [{i.kind}] {i.question}")
                L.append(f"      predicted for: {i.predicted_for}  "
                         f"(P(complainant) = {i.probability:.3f}, margin {i.margin:+.3f}){mark}")
                for b in i.basis:
                    L.append(f"      basis: {b}")
                for c in i.caveats:
                    L.append(f"      caveat: {c}")

        rf = self.relief_forecast
        if rf.relief_heads or rf.principal_claimed_inr:
            L.append(f"\nRELIEF FORECAST (conditional on {rf.conditional_on})")
            if rf.principal_claimed_inr:
                L.append(f"  principal claimed: Rs.{rf.principal_claimed_inr:,}")
            if rf.relief_heads:
                L.append(f"  heads supported by precedent: {', '.join(rf.relief_heads)}")
            if rf.claimed_heads_without_precedent:
                L.append("  claimed but not granted in any verified precedent: "
                         f"{', '.join(rf.claimed_heads_without_precedent)}")
            if rf.typical_interest_rate_pct is not None:
                L.append(f"  typical interest in comparable decisions: "
                         f"{rf.typical_interest_rate_pct}% p.a. "
                         f"({', '.join(rf.interest_rate_basis)})")
            if rf.note:
                L.append(f"  note: {rf.note}")

        if self.counterfactuals:
            L.append(f"\nWHAT WOULD CHANGE THIS")
            for c in self.counterfactuals:
                flag = "  -> FLIPS THE PREDICTION" if c.flips_prediction else ""
                L.append(f"  - {c.change}")
                L.append(f"      P becomes {c.revised_probability:.3f} "
                         f"({c.revised_disposition}){flag}")

        L.append(f"\nHOW FAR THIS CAN BE TRUSTED")
        L.append(f"  evidence quality {self.evidence_quality:.3f}  " + "  ".join(
            f"{k}={v:.2f}" for k, v in self.evidence_components.items()))
        if self.evidential_gaps:
            L.append(f"  gaps the explanation must concede ({len(self.evidential_gaps)}):")
            for g in self.evidential_gaps[:8]:
                L.append(f"    - {g}")
        if self.excluded_from_prior:
            L.append("  authority shown but excluded from the prior (failed verification):")
            for e in self.excluded_from_prior[:8]:
                L.append(f"    - {e}")
        for w in self.warnings:
            L.append(f"WARNING: {w}")
        return "\n".join(L)


# --------------------------------------------------------------------------
# The agent
# --------------------------------------------------------------------------
LLM_SYSTEM = (
    "You are assisting a legal judgment prediction system for Indian consumer disputes "
    "(Consumer Protection Act, 2019; NCDRC practice). You are given the issues framed in a "
    "pending case, each side's argued strength, and the outcome of comparable verified "
    "precedents. Assess only the DIRECTION and STRENGTH of the merits, on a scale from -1 "
    "(the opposite party should succeed) to +1 (the complainant should succeed). "
    'Respond ONLY with JSON: {"merits_direction": <float -1..1>, "rationale": "<one sentence>"}. '
    "Do not invent facts, authorities or provisions beyond those supplied."
)


class PredictionAgent(BaseAgent):
    """
    Agent 5: predicts the disposition, with a calibrated confidence and a
    fully decomposed account of how it got there.

    Usage inside the pipeline (LangGraph-compatible, same contract as Agents 1-4):
        state = FactExtractionAgent().run(state)
        state = StatuteRetrievalAgent(conn=conn).run(state)
        state = PrecedentRetrievalAgent(conn=conn).run(state)
        state = ArgumentAnalysisAgent().run(state)
        state = PredictionAgent().run(state)          # -> state["prediction"]
    """
    name = "prediction_agent"

    def __init__(self, llm=None, abstain_threshold: float = DEFAULT_ABSTAIN_THRESHOLD,
                 neutral_prior: float = NEUTRAL_PRIOR, use_precedent_prior: bool = True,
                 count_unverified_prior: bool = False):
        """
        abstain_threshold: below this confidence the agent returns
            `indeterminate` rather than a disposition. Set to 0.0 to force a
            prediction on every case - useful for a coverage/accuracy trade-off
            curve, wrong for production use.
        use_precedent_prior: set False to ablate the precedential prior and
            predict from the argument map alone.
        count_unverified_prior: let unverified precedents into the prior.
            Default False (verification-first); available for ablation.
        """
        self.llm = llm
        self.abstain_threshold = abstain_threshold
        self.neutral_prior = neutral_prior
        self.use_precedent_prior = use_precedent_prior
        self.count_unverified_prior = count_unverified_prior
        self.llm_error: str | None = None

    # ---- pipeline interface -------------------------------------------------
    def run(self, state: dict) -> dict:
        pred = self.predict(
            argument_map=state.get("argument_map"),
            precedent_bundle=state.get("precedent_bundle"),
            statute_bundle=state.get("statute_bundle"),
            fact_sheet=state.get("fact_sheet"),
        )
        state["prediction"] = pred
        state["prediction_summary"] = pred.for_explanation()
        return state

    # ---- main entry ---------------------------------------------------------
    def predict(self, argument_map=None, precedent_bundle=None,
                statute_bundle=None, fact_sheet=None) -> Prediction:
        p = Prediction()
        p.inputs_used = [n for n, v in (("argument_map", argument_map),
                                        ("precedent_bundle", precedent_bundle),
                                        ("statute_bundle", statute_bundle),
                                        ("fact_sheet", fact_sheet)) if v is not None]
        p.ranked_by = "rules+llm" if self.llm is not None else "rules"

        if argument_map is None or not argument_map.issues:
            p.abstained = True
            p.decided_at = "abstained"
            p.abstention_reasons.append(
                "No argument map with framed issues - run the Argument Analysis Agent first. "
                "Nothing can be predicted from an unanalysed case file.")
            p.warnings.append("No argument map supplied; no prediction attempted.")
            return p

        if fact_sheet is not None and not fact_sheet.outcome_removed:
            p.leakage_suspect = True
            p.warnings.append(
                "Agent 1 reports the operative order was NOT stripped from this case file. The "
                "upstream bundles may carry the real outcome, so this prediction is unsafe to "
                "use as a backtest result.")
        if precedent_bundle is None:
            p.warnings.append(
                "No precedent bundle - predicting without a precedential prior; confidence is "
                "reduced accordingly. Run the Precedent Retrieval Agent for a grounded prior.")
        if statute_bundle is None:
            p.warnings.append(
                "No statute bundle - no element coverage and no computed limitation or "
                "jurisdiction finding; threshold issues rest on the pleadings alone.")

        # --- 1. the prior, from verified comparable decisions
        self._set_prior(p, precedent_bundle)

        # --- 2. threshold gate: decided first, as a tribunal decides them
        issue_preds, gate = self._threshold_gate(argument_map, statute_bundle, fact_sheet)
        p.issue_predictions = issue_preds
        p.threshold_risk = round(gate["risk"], 4)
        p.dispositive_grounds = gate["grounds"]

        # --- 3. the merits update, in log-odds
        merits_logit = self._merits_logit(p, argument_map, statute_bundle)
        p.posterior_logit = round(merits_logit, 4)
        p_merits = sigmoid(merits_logit)

        # --- 4. combine. A threshold issue that is likely fatal disposes of the
        # case; one that is merely live still discounts the merits, because a
        # complaint that may not be entertained is not a complaint that wins.
        if gate["risk"] >= DISPOSITIVE_GATE:
            p.decided_at = "threshold"
            p.probability_complainant_succeeds = round(p_merits * (1.0 - gate["risk"]), 4)
        else:
            p.decided_at = "merits"
            discount = 1.0 - RESIDUAL_THRESHOLD_WEIGHT * gate["risk"]
            p.probability_complainant_succeeds = round(p_merits * max(0.0, discount), 4)

        # --- 5. fill in the merits issue predictions now the posterior is known
        self._predict_merits_issues(p, argument_map)

        # --- 6. evidence quality, then calibrated confidence
        self._score_evidence(p, argument_map, precedent_bundle, statute_bundle)
        self._set_disposition(p, precedent_bundle, argument_map)

        # --- 7. what follows, and what would change it
        p.relief_forecast = self._forecast_relief(fact_sheet, precedent_bundle, argument_map)
        p.counterfactuals = self._counterfactuals(p, argument_map, gate, merits_logit,
                                                  precedent_bundle)

        p.evidential_gaps = list(argument_map.evidential_gaps)
        p.excluded_from_prior = sorted(dict.fromkeys(p.excluded_from_prior))
        return p

    # ---- 1. prior ------------------------------------------------------------
    def _set_prior(self, p: Prediction, precedent_bundle):
        """
        The prior is the authority-weighted rate at which VERIFIED comparable
        decisions went for the complainant, shrunk toward neutral by how few of
        them there are.

        Authority weighting matters: three District Forum orders should not
        outweigh one NCDRC decision, and Agent 3 has already computed that rate.
        Shrinkage matters more: without it, a corpus of two concordant cases
        would hand the model a prior of 1.0 and every subsequent step would be
        arguing with a certainty it never earned.
        """
        if not self.use_precedent_prior or precedent_bundle is None:
            p.prior = self.neutral_prior
            p.prior_basis = ("no precedential prior used (ablation)" if not self.use_precedent_prior
                             else "no precedent bundle supplied; neutral prior")
            return

        usable = [m for m in precedent_bundle.matches
                  if (m.verified or self.count_unverified_prior) and m.outcome.label != "unknown"]
        for m in precedent_bundle.matches:
            if m.verified or self.count_unverified_prior:
                continue
            p.excluded_from_prior.append(
                f"{m.case_number or 'unnamed precedent'} ({m.forum}) - "
                f"{m.verification_reason or 'not verified against the curated corpus'}")

        if not usable:
            p.prior = self.neutral_prior
            p.prior_basis = ("no verified precedent with a readable outcome; neutral prior "
                             f"({len(precedent_bundle.matches)} match(es) excluded)")
            return

        if self.count_unverified_prior:
            favourable = [m for m in usable if m.outcome.favours == "complainant"]
            total_w = sum(authority_weight(m.forum) for m in usable) or 1.0
            raw = sum(authority_weight(m.forum) for m in favourable) / total_w
        else:
            raw = (precedent_bundle.authority_weighted_success_rate
                   if precedent_bundle.authority_weighted_success_rate is not None
                   else precedent_bundle.complainant_success_rate)

        p.prior = round(shrink(raw, len(usable), self.neutral_prior), 4)
        p.prior_basis = (
            f"{len(usable)} verified comparable decision(s), authority-weighted success rate "
            f"{raw if raw is not None else 'n/a'}, shrunk toward {self.neutral_prior} "
            f"(pseudo-count {PRIOR_PSEUDO_COUNT}) because the corpus is small")

    # ---- 2. threshold gate ---------------------------------------------------
    def _threshold_gate(self, amap, statute_bundle, fact_sheet) -> tuple[list, dict]:
        """
        Decide the threshold issues first, and let a fatal one end the case.

        This ordering is the whole point. A tribunal that holds a complaint
        time-barred never reaches deficiency, so a model that averages
        limitation into an overall merits score will confidently predict a win
        in a case that is going to be thrown out at the door. Threshold issues
        are therefore evaluated separately and can short-circuit everything
        downstream.
        """
        preds, grounds, risks = [], [], []

        for issue in amap.issues:
            if issue.kind != "threshold":
                continue
            comp, opp = issue.complainant.strength, issue.opposite_party.strength
            basis, caveats = [], []

            if issue.leaning == "unsupported":
                p_fail = UNSUPPORTED_THRESHOLD_RISK
                basis.append("neither side argued this threshold; tribunals do not dismiss for "
                             "want of a plea nobody took, so the risk is treated as small")
            else:
                margin = comp - opp
                if issue.one_sided_record:
                    margin *= ONE_SIDED_DISCOUNT
                    caveats.append(
                        "one-sided pleading record - the margin is halved because the answering "
                        "argument is likely in the tribunal's voice and was excluded upstream")
                p_fail = sigmoid(-THRESHOLD_GAIN * margin)
                basis.append(f"argued strengths: complainant {comp:.3f} v opposite party "
                             f"{opp:.3f} (margin {margin:+.3f})")

            # Agent 2's computed findings are harder evidence than argued
            # strengths - they come from dates and amounts, not from prose - so
            # they set a floor rather than being averaged in.
            if statute_bundle is not None:
                if issue.id == "limitation" and statute_bundle.limitation.status == "prima_facie_barred":
                    floor = (LIMITATION_BARRED_FLOOR_IF_CONTINUING
                             if statute_bundle.limitation.continuing_cause_indicated
                             else LIMITATION_BARRED_FLOOR)
                    if floor > p_fail:
                        p_fail = floor
                    basis.append(f"Agent 2: prima facie barred - {statute_bundle.limitation.note}")
                    if statute_bundle.limitation.continuing_cause_indicated:
                        caveats.append("a continuing cause of action is pleaded, which if accepted "
                                       "defeats the limitation plea - the floor is reduced accordingly")
                if issue.id == "limitation" and statute_bundle.limitation.status == "within_time":
                    p_fail = min(p_fail, 0.25)
                    basis.append(f"Agent 2: within time - {statute_bundle.limitation.note}")
                if issue.id == "jurisdiction" and statute_bundle.jurisdiction.determinate:
                    j = statute_bundle.jurisdiction
                    seat = (fact_sheet.forum if fact_sheet else None)
                    basis.append(f"Agent 2: Rs.{j.amount_in_issue_inr:,} -> {j.forum}")
                    if seat and j.forum and seat.strip().lower() != j.forum.strip().lower():
                        p_fail = max(p_fail, FORUM_MISMATCH_FLOOR)
                        caveats.append(
                            f"pecuniary computation points to the {j.forum} but the case is before "
                            f"the {seat} - a return for want of jurisdiction is a live risk")
                    elif seat:
                        p_fail = min(p_fail, 0.25)

            p_fail = min(max(p_fail, 0.0), 1.0)
            dispositive = p_fail >= DISPOSITIVE_GATE
            preds.append(IssuePrediction(
                issue_id=issue.id, question=issue.question, kind=issue.kind,
                predicted_for=("opposite_party" if p_fail > 0.55 else
                               "complainant" if p_fail < 0.45 else "too_close_to_call"),
                probability=round(1.0 - p_fail, 4), margin=round(issue.margin, 4),
                dispositive=dispositive, one_sided_record=issue.one_sided_record,
                basis=basis, caveats=caveats))
            risks.append(p_fail)
            if dispositive:
                grounds.append({"issue_id": issue.id, "question": issue.question,
                                "probability": round(p_fail, 4), "basis": "; ".join(basis)})

        # The risk of the case ending at the door is the risk of the WORST
        # threshold issue, not the average: they are alternative fatal grounds,
        # and passing one does not cure another.
        return preds, {"risk": max(risks) if risks else 0.0, "grounds": grounds}

    # ---- 3. merits update ----------------------------------------------------
    def _merits_logit(self, p: Prediction, amap, statute_bundle) -> float:
        """
        Move the prior's log-odds by this case's own evidence, recording every
        term so the result decomposes exactly.
        """
        base = logit(p.prior)
        drivers: list[Driver] = []

        # (a) the merits balance Agent 4 computed, already verification-filtered
        mb = amap.merits_balance
        c = POSTERIOR_WEIGHTS["merits_balance"] * mb
        drivers.append(Driver(
            name="merits_balance", value=round(mb, 4), logit_contribution=round(c, 4),
            direction=self._dir(mb),
            explanation=("net argued strength across the merits issues, counting only support "
                         "that passed verification"),
            grounded_in=[i.id for i in amap.issues if i.kind == "merits"]))

        # (b) statutory element coverage: has the complainant actually made out
        # the ingredients of the provision relied on? Centred at 0.5, so
        # half-established elements move nothing.
        cov, cites = self._element_coverage(statute_bundle)
        if cov is not None:
            v = (cov - 0.5) * 2.0
            c = POSTERIOR_WEIGHTS["element_coverage"] * v
            drivers.append(Driver(
                name="element_coverage", value=round(v, 4), logit_contribution=round(c, 4),
                direction=self._dir(v),
                explanation=(f"{cov:.0%} of the statutory elements of the verified operative "
                             "provisions are established on the facts"),
                grounded_in=cites))

        # (c) unrebutted contentions. An averment the other side never answered
        # carries real weight in consumer adjudication, where the pleadings are
        # often one-sided and an uncontroverted assertion may be accepted.
        adv = self._unrebutted_advantage(amap)
        c = POSTERIOR_WEIGHTS["unrebutted_advantage"] * adv
        drivers.append(Driver(
            name="unrebutted_advantage", value=round(adv, 4), logit_contribution=round(c, 4),
            direction=self._dir(adv),
            explanation="net share of contentions the other side left unanswered",
            grounded_in=(amap.unrebutted_by_party.get("complainant", [])[:5])))

        # (d) evidential gaps. Unsatisfied elements and unsupported contentions
        # cut against the party bearing the burden, which is the complainant.
        n_gaps = min(len(amap.evidential_gaps), GAP_PENALTY_CAP)
        if n_gaps:
            c = POSTERIOR_WEIGHTS["gap_penalty"] * n_gaps
            drivers.append(Driver(
                name="evidential_gaps", value=float(n_gaps), logit_contribution=round(c, 4),
                direction="opposite_party",
                explanation=(f"{len(amap.evidential_gaps)} gap(s) in the complainant's case "
                             f"(counted up to {GAP_PENALTY_CAP}); the burden is on the complainant"),
                grounded_in=amap.evidential_gaps[:3]))

        # (e) optional, bounded LLM opinion
        llm_shift = self._llm_shift(amap, p)
        if llm_shift is not None:
            drivers.append(llm_shift)

        p.drivers = drivers
        return base + sum(d.logit_contribution for d in drivers)

    @staticmethod
    def _dir(v: float) -> str:
        return "complainant" if v > 1e-9 else "opposite_party" if v < -1e-9 else "neutral"

    @staticmethod
    def _element_coverage(statute_bundle) -> tuple[float | None, list]:
        """
        Mean element coverage over the VERIFIED operative provisions.

        Definitions and jurisdiction clauses are excluded: the question is
        whether the complainant has made out the ingredients of the provision
        that creates the liability, not whether a definition exists.
        """
        if statute_bundle is None:
            return None, []
        ops = [m for m in statute_bundle.matches
               if m.verified and m.admissible_as_authority and m.elements]
        if not ops:
            return None, []
        return (round(sum(m.element_coverage for m in ops) / len(ops), 4),
                [m.citation for m in ops][:5])

    @staticmethod
    def _unrebutted_advantage(amap) -> float:
        """Net unrebutted share: +1 all the complainant's, -1 all the OP's."""
        c = len(amap.unrebutted_by_party.get("complainant", []))
        o = len(amap.unrebutted_by_party.get("opposite_party", []))
        return (c - o) / (c + o) if (c + o) else 0.0

    def _llm_shift(self, amap, p: Prediction) -> Driver | None:
        """
        Admit an LLM's view of the merits as a BOUNDED adjustment.

        The bound is the point. An unbounded LLM opinion would make every
        preceding step decorative; clamped to +/-LLM_MAX_LOGIT_SHIFT it can
        shade a borderline case without ever overturning the grounded evidence,
        and it is reported as its own driver so a reader can see exactly how
        much of the prediction is model opinion rather than record.
        """
        if self.llm is None:
            return None
        try:
            issues = "\n".join(
                f"{i.id} [{i.kind}] {i.question} | complainant {i.complainant.strength:.2f} "
                f"v opposite party {i.opposite_party.strength:.2f} | leaning {i.leaning}"
                for i in amap.issues)
            reply = self.llm.complete_json(
                LLM_SYSTEM,
                f"ISSUES\n{issues}\n\n"
                f"MERITS BALANCE: {amap.merits_balance}\n"
                f"THRESHOLD RISK: {amap.threshold_risk}\n"
                f"PRECEDENTIAL PRIOR: {p.prior}\n"
                f"EVIDENTIAL GAPS: {amap.evidential_gaps[:5]}")
            direction = float(reply["merits_direction"])
            direction = min(max(direction, -1.0), 1.0)
            shift = LLM_MAX_LOGIT_SHIFT * direction
            return Driver(
                name="llm_merits_assessment", value=round(direction, 4),
                logit_contribution=round(shift, 4), direction=self._dir(direction),
                explanation=(str(reply.get("rationale", ""))[:240] +
                             f" [bounded to +/-{LLM_MAX_LOGIT_SHIFT} log-odds]"))
        except Exception as e:  # noqa: BLE001 - any failure => rules only
            self.llm_error = str(e)
            p.warnings.append(f"LLM merits assessment unavailable ({e}); prediction is rule-based.")
            p.ranked_by = "rules"
            return None

    # ---- 5. merits issue predictions ----------------------------------------
    @staticmethod
    def _predict_merits_issues(p: Prediction, amap):
        """
        A per-issue finding for every merits and relief issue, so the
        Explanation Agent can write an IRAC block per issue rather than one
        undifferentiated verdict.
        """
        for issue in amap.issues:
            if issue.kind == "threshold":
                continue
            comp, opp = issue.complainant.strength, issue.opposite_party.strength
            margin = comp - opp
            if issue.one_sided_record:
                margin *= ONE_SIDED_DISCOUNT
            prob = sigmoid(THRESHOLD_GAIN * margin) if (comp or opp) else 0.5
            basis = [f"argued strengths: complainant {comp:.3f} v opposite party {opp:.3f}"]
            basis += [n for n in issue.notes[:2]]
            caveats = []
            if issue.one_sided_record:
                caveats.append("one-sided pleading record - margin discounted")
            if issue.leaning == "unsupported":
                caveats.append("neither side made a scored argument on this issue")
            p.issue_predictions.append(IssuePrediction(
                issue_id=issue.id, question=issue.question, kind=issue.kind,
                predicted_for=("complainant" if prob > 0.55 else
                               "opposite_party" if prob < 0.45 else "too_close_to_call"),
                probability=round(prob, 4), margin=round(issue.margin, 4),
                one_sided_record=issue.one_sided_record, basis=basis, caveats=caveats))

    # ---- 6. calibration and disposition -------------------------------------
    @staticmethod
    def _score_evidence(p: Prediction, amap, precedent_bundle, statute_bundle):
        """
        How much evidence stands behind the number.

        This is what separates a defensible 0.78 from an arithmetically
        identical but worthless one. Directional certainty is cheap - two
        sentences and one precedent can produce a lopsided score. Confidence
        should track the RECORD, so it is scaled by this.
        """
        n_verified = len([m for m in precedent_bundle.matches if m.verified]) \
            if precedent_bundle else 0
        n_contentions = sum(len(i.complainant.contentions) + len(i.opposite_party.contentions)
                            for i in amap.issues)
        n_statutes = len([m for m in statute_bundle.matches
                          if m.verified and m.admissible_as_authority]) if statute_bundle else 0

        comp = {
            "verified_precedents": min(1.0, n_verified / VERIFIED_PRECEDENT_SATURATION),
            "grounded_contentions": min(1.0, n_contentions / CONTENTION_SATURATION),
            "statutory_grounding": min(1.0, n_statutes / 3.0),
            "input_completeness": len(p.inputs_used) / 4.0,
        }
        p.evidence_components = {k: round(v, 3) for k, v in comp.items()}
        p.evidence_quality = round(sum(EVIDENCE_WEIGHTS[k] * v for k, v in comp.items()), 4)

    def _set_disposition(self, p: Prediction, precedent_bundle, amap):
        """
        Turn the probability into a disposition, a confidence, and - where the
        record will not bear a prediction - an abstention.
        """
        prob = p.probability_complainant_succeeds

        # Directional certainty, scaled by the record behind it, then capped.
        directional = abs(prob - 0.5) * 2.0
        p.confidence = round(min(CONFIDENCE_CEILING, directional * p.evidence_quality), 4)

        if p.confidence < self.abstain_threshold:
            p.abstained = True
            p.disposition = INDETERMINATE
            p.favours = "unknown"
            p.decided_at = "abstained"
            p.abstention_reasons = self._abstention_reasons(p, directional)
            p.disposition_probabilities = self._split(prob, precedent_bundle, amap)
            return

        # Split a complainant win between full and partial relief.
        dist = self._split(prob, precedent_bundle, amap)
        p.disposition_probabilities = dist
        p.disposition = max(dist, key=dist.get)
        p.favours = "complainant" if p.disposition in COMPLAINANT_FAVOURABLE else "opposite_party"

    def _abstention_reasons(self, p: Prediction, directional: float) -> list:
        reasons = []
        if directional < 0.25:
            reasons.append(
                f"The case is genuinely close (P = {p.probability_complainant_succeeds:.3f}); the "
                "evidence does not favour either side clearly enough to call.")
        if p.evidence_quality < 0.45:
            weak = [k for k, v in p.evidence_components.items() if v < 0.5]
            reasons.append(
                f"The record is too thin to support a prediction (evidence quality "
                f"{p.evidence_quality:.2f}); weakest: {', '.join(weak) or 'n/a'}.")
        if "precedent_bundle" not in p.inputs_used:
            reasons.append("No precedential prior: the prediction would rest on the pleadings alone.")
        if "statute_bundle" not in p.inputs_used:
            reasons.append("No statutory element analysis: whether the ingredients of the "
                           "provision are made out was never assessed.")
        reasons.append(
            f"Confidence {p.confidence:.3f} is below the abstention threshold "
            f"{self.abstain_threshold}. Returning `indeterminate` is the correct output here - a "
            "prediction this weakly supported would mislead a user who relied on it.")
        return reasons

    @staticmethod
    def _split(prob: float, precedent_bundle, amap) -> dict:
        """
        Split the complainant's probability of success between full and partial
        relief, using the corpus where it can support the estimate.

        Predicting `allowed` for every complainant win is a real accuracy loss:
        consumer commissions frequently grant the principal with interest while
        trimming the compensation claimed, which the corpus records as partly
        allowed. Where the complainant's relief case is itself well made out,
        the split tilts back toward full allowance.
        """
        share = PARTLY_ALLOWED_SHARE
        basis_n = 0
        if precedent_bundle is not None:
            dist = precedent_bundle.outcome_distribution or {}
            full = dist.get(ALLOWED, 0) + dist.get("disposed_with_directions", 0)
            part = dist.get(PARTLY_ALLOWED, 0)
            basis_n = full + part
            if basis_n:
                share = shrink(part / basis_n, basis_n, PARTLY_ALLOWED_SHARE)

        # A strongly made-out relief issue pulls toward relief as prayed.
        relief = next((i for i in amap.issues if i.kind == "relief"), None)
        if relief is not None and relief.complainant.strength > RELIEF_STRENGTH_PIVOT:
            share *= 0.75

        share = min(max(share, 0.0), 1.0)
        return {
            ALLOWED: round(prob * (1.0 - share), 4),
            PARTLY_ALLOWED: round(prob * share, 4),
            DISMISSED: round(1.0 - prob, 4),
        }

    # ---- 7. relief and counterfactuals --------------------------------------
    @staticmethod
    def _forecast_relief(fact_sheet, precedent_bundle, amap) -> ReliefForecast:
        """
        What would be awarded if the complaint succeeds - grounded twice.

        A head is forecast only if the complainant claimed it AND a verified
        precedent granted it. That double grounding is what keeps this a
        forecast rather than a wish list: a system that predicted compensation
        for mental agony in every case, because complainants always ask for it,
        would be reporting the pleadings back to the user as though they were
        evidence.
        """
        rf = ReliefForecast()
        if fact_sheet is None:
            rf.note = "No fact sheet - relief not forecast."
            return rf

        claimed: set[str] = set()
        for f in fact_sheet.claims:
            for h in ("refund", "interest", "compensation", "replacement", "possession"):
                if h in f.text.lower():
                    claimed.add(h)
            if "cost" in f.text.lower():
                claimed.add("litigation_cost")
        rf.principal_claimed_inr = fact_sheet.total_amount_paid_inr

        granted: set[str] = set()
        rates, amounts = [], []
        if precedent_bundle is not None:
            for m in precedent_bundle.matches:
                if not m.verified or m.outcome.favours != "complainant":
                    continue
                granted.update(m.outcome.relief_types)
                rates.extend(m.outcome.interest_rates)
                amounts.extend(m.outcome.amounts_inr)
                if m.outcome.interest_rates:
                    rf.interest_rate_basis.append(f"{m.case_number} ({m.forum})")

        rf.relief_heads = sorted(claimed & granted)
        rf.claimed_heads_without_precedent = sorted(claimed - granted)
        rf.typical_interest_rate_pct = median_or_none(rates)
        rf.precedent_amounts_inr = sorted(set(amounts))[:8]
        rf.interest_rate_basis = rf.interest_rate_basis[:4]

        if not precedent_bundle or not granted:
            rf.note = ("No verified precedent granting relief was available, so no head could be "
                       "grounded in comparable practice; the claimed heads are listed unverified.")
        else:
            rf.note = ("Indicative only. Heads are those both claimed in this case and granted in "
                       "verified comparable decisions; amounts reflect the corpus, not an award, "
                       "and quantum is a matter for the Commission.")
        return rf

    def _counterfactuals(self, p: Prediction, amap, gate, merits_logit: float,
                         precedent_bundle=None) -> list:
        """
        What would have to change for the prediction to change.

        This is the actionable half of transparency. "Dismissal, 71% confident"
        tells a user nothing they can do; "clear the limitation objection and
        this becomes a 68% partial allowance" tells them what the case turns on.
        Each scenario is recomputed through the same model, so these are real
        model outputs rather than commentary.
        """
        out = []
        current = p.disposition
        p_merits = sigmoid(merits_logit)

        def disposition_for(prob: float) -> str:
            # The SAME split the live prediction used, so a counterfactual is a
            # real model output under a changed input rather than a second,
            # differently-parameterised model that happens to agree.
            d = self._split(prob, precedent_bundle, amap)
            return max(d, key=d.get)

        # (a) the threshold objection is answered
        if gate["risk"] > 0.05:
            revised = round(p_merits, 4)
            out.append(Counterfactual(
                change=("the threshold objection(s) are answered in the complainant's favour "
                        f"({', '.join(g['issue_id'] for g in gate['grounds']) or 'threshold risk'} "
                        "cleared)"),
                revised_probability=revised, revised_disposition=disposition_for(revised),
                flips_prediction=disposition_for(revised) != current))

        # (b) the unsatisfied statutory elements are established
        gap_driver = next((d for d in p.drivers if d.name == "evidential_gaps"), None)
        if gap_driver is not None:
            revised_logit = merits_logit - gap_driver.logit_contribution
            revised = round(sigmoid(revised_logit) * (1.0 - gate["risk"]), 4)
            out.append(Counterfactual(
                change=(f"the {int(gap_driver.value)} unsatisfied statutory element(s) / "
                        "unsupported contention(s) are made out on evidence"),
                revised_probability=revised, revised_disposition=disposition_for(revised),
                flips_prediction=disposition_for(revised) != current))

        # (c) the opposite party's case is fully rebutted on the merits
        mb_driver = next((d for d in p.drivers if d.name == "merits_balance"), None)
        if mb_driver is not None and amap.merits_balance < 0.9:
            revised_logit = (merits_logit - mb_driver.logit_contribution
                             + POSTERIOR_WEIGHTS["merits_balance"] * 1.0)
            revised = round(sigmoid(revised_logit) * (1.0 - gate["risk"]), 4)
            out.append(Counterfactual(
                change="the complainant carries every merits issue outright",
                revised_probability=revised, revised_disposition=disposition_for(revised),
                flips_prediction=disposition_for(revised) != current))

        # (d) the precedential prior is withdrawn - how much of this is the corpus?
        if p.prior != self.neutral_prior:
            revised_logit = merits_logit - logit(p.prior) + logit(self.neutral_prior)
            revised = round(sigmoid(revised_logit) * (1.0 - gate["risk"]), 4)
            out.append(Counterfactual(
                change="the precedential prior is set aside (neutral prior)",
                revised_probability=revised, revised_disposition=disposition_for(revised),
                flips_prediction=disposition_for(revised) != current))
        return out


# --------------------------------------------------------------------------
# Backtesting helper.
#
# Kept here rather than in the script so the evaluation metric is versioned
# with the model it scores.
# --------------------------------------------------------------------------
def score_against_actual(pred: Prediction, actual_label: str) -> dict:
    """
    Compare a prediction with the real disposition of a decided judgment.

    Two notions of correctness are reported, because they answer different
    questions. `direction_correct` asks whether the system got the WINNER
    right - which is what a user cares about. `exact_correct` asks whether it
    got the precise disposition, where confusing `allowed` with
    `partly_allowed` counts as a miss. Direction accuracy is the headline
    figure; reporting only the exact figure would understate the system, and
    reporting only direction would overstate it.

    `brier` is the squared error of the probability against the realised
    outcome - the calibration measure. A model that is right but overconfident
    is penalised here, which is exactly what should happen to a legal
    prediction system.
    """
    actual_favours = "complainant" if actual_label in COMPLAINANT_FAVOURABLE else "opposite_party"
    y = 1.0 if actual_favours == "complainant" else 0.0
    return {
        "predicted": pred.disposition,
        "actual": actual_label,
        "abstained": pred.abstained,
        "direction_correct": (None if pred.abstained else pred.favours == actual_favours),
        "exact_correct": (None if pred.abstained else pred.disposition == actual_label),
        "probability": pred.probability_complainant_succeeds,
        "confidence": pred.confidence,
        "brier": round((pred.probability_complainant_succeeds - y) ** 2, 4),
        "leakage_suspect": pred.leakage_suspect,
    }
