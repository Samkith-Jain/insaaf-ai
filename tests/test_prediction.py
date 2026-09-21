"""
Run:  python -m unittest tests.test_prediction -v

The precedent corpus reused here is the SYNTHETIC fixture set from
tests/test_precedent_retrieval.py - not real decisions. Nothing in this file
measures predictive accuracy; these tests check that the model behaves the way
it is documented to behave (ordering, calibration, abstention, leakage,
verification-first), which is what can honestly be tested on a fixture corpus.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.argument_analysis import ArgumentAnalysisAgent
from agents.fact_extraction import FactExtractionAgent
from agents.precedent_retrieval import PrecedentRetrievalAgent
from agents.prediction import (
    ALLOWED, DISMISSED, INDETERMINATE, PARTLY_ALLOWED, CONFIDENCE_CEILING,
    DISPOSITIVE_GATE, EVIDENCE_WEIGHTS, NEUTRAL_PRIOR, POSTERIOR_WEIGHTS,
    Prediction, PredictionAgent, logit, median_or_none, score_against_actual,
    shrink, sigmoid,
)
from agents.statute_retrieval import StatuteRetrievalAgent
from tests.test_argument_analysis import BUILDER_CASE, UNCONTESTED_CASE
from tests.test_precedent_retrieval import fresh_db


def full_state(case_text, query="", conn=None, with_precedents=True, agent=None):
    """Run Agents 1 -> 2 -> 3 -> 4 -> 5 the way the pipeline does."""
    conn = conn or fresh_db()
    state = {"case_text": case_text, "query": query}
    state = FactExtractionAgent().run(state)
    state = StatuteRetrievalAgent(conn=conn).run(state)
    if with_precedents:
        state = PrecedentRetrievalAgent(conn=conn).run(state)
    state = ArgumentAnalysisAgent().run(state)
    return (agent or PredictionAgent()).run(state)


class StubLLM:
    """An LLM that always insists the opposite party should win."""
    def __init__(self, direction=-1.0, fail=False):
        self.direction, self.fail = direction, fail

    def complete_json(self, system, user):
        if self.fail:
            raise RuntimeError("no network")
        return {"merits_direction": self.direction, "rationale": "stub"}


# --------------------------------------------------------------------------
class TestNumerics(unittest.TestCase):
    def test_sigmoid_logit_roundtrip(self):
        for p in (0.01, 0.25, 0.5, 0.75, 0.99):
            self.assertAlmostEqual(sigmoid(logit(p)), p, places=6)

    def test_sigmoid_is_stable_at_extremes(self):
        self.assertAlmostEqual(sigmoid(-1000), 0.0, places=9)
        self.assertAlmostEqual(sigmoid(1000), 1.0, places=9)

    def test_logit_clamps_degenerate_probabilities(self):
        self.assertTrue(math_isfinite(logit(0.0)) and math_isfinite(logit(1.0)))

    def test_shrinkage_pulls_small_samples_toward_neutral(self):
        # Two concordant cases must not produce certainty.
        self.assertLess(shrink(1.0, 2), 0.9)
        self.assertGreater(shrink(1.0, 2), NEUTRAL_PRIOR)

    def test_shrinkage_converges_on_large_samples(self):
        self.assertAlmostEqual(shrink(0.8, 500), 0.8, places=2)

    def test_shrinkage_with_no_observations_is_neutral(self):
        self.assertEqual(shrink(None, 0), NEUTRAL_PRIOR)
        self.assertEqual(shrink(0.9, 0), NEUTRAL_PRIOR)

    def test_median_or_none(self):
        self.assertEqual(median_or_none([6, 8, 10]), 8.0)
        self.assertIsNone(median_or_none([]))
        self.assertIsNone(median_or_none([None, "x"]))


def math_isfinite(x):
    import math
    return math.isfinite(x)


# --------------------------------------------------------------------------
class TestGuards(unittest.TestCase):
    def test_no_argument_map_abstains_rather_than_guessing(self):
        p = PredictionAgent().predict(argument_map=None)
        self.assertTrue(p.abstained)
        self.assertEqual(p.disposition, INDETERMINATE)
        self.assertEqual(p.decided_at, "abstained")
        self.assertTrue(p.abstention_reasons)

    def test_empty_issue_list_abstains(self):
        from agents.argument_analysis import ArgumentMap
        p = PredictionAgent().predict(argument_map=ArgumentMap())
        self.assertTrue(p.abstained)

    def test_confidence_never_reaches_certainty(self):
        state = full_state(UNCONTESTED_CASE)
        self.assertLessEqual(state["prediction"].confidence, CONFIDENCE_CEILING)

    def test_probability_stays_in_range(self):
        for case in (BUILDER_CASE, UNCONTESTED_CASE):
            p = full_state(case)["prediction"]
            self.assertGreaterEqual(p.probability_complainant_succeeds, 0.0)
            self.assertLessEqual(p.probability_complainant_succeeds, 1.0)

    def test_weights_are_documented_and_finite(self):
        self.assertAlmostEqual(sum(EVIDENCE_WEIGHTS.values()), 1.0, places=9)
        for k, v in POSTERIOR_WEIGHTS.items():
            self.assertIsInstance(v, float, k)


# --------------------------------------------------------------------------
class TestThresholdGate(unittest.TestCase):
    """Threshold issues must be able to dispose of a case before the merits."""

    def setUp(self):
        self.p = full_state(BUILDER_CASE)["prediction"]

    def test_limitation_plea_is_gated_not_averaged(self):
        self.assertEqual(self.p.decided_at, "threshold")
        self.assertTrue(self.p.dispositive_grounds)

    def test_strong_merits_do_not_override_a_fatal_threshold(self):
        # The builder case has a strong merits balance and is still predicted
        # to fail - that is the whole point of the gate.
        merits_logit = self.p.posterior_logit
        self.assertGreater(sigmoid(merits_logit), 0.5)
        self.assertLess(self.p.probability_complainant_succeeds, 0.5)
        self.assertEqual(self.p.disposition, DISMISSED)

    def test_dispositive_ground_names_the_issue_and_its_basis(self):
        g = self.p.dispositive_grounds[0]
        self.assertIn("issue_id", g)
        self.assertGreaterEqual(g["probability"], DISPOSITIVE_GATE)
        self.assertTrue(g["basis"])

    def test_threshold_risk_is_the_worst_issue_not_the_mean(self):
        thresholds = [i for i in self.p.issue_predictions if i.kind == "threshold"]
        worst = max(1.0 - i.probability for i in thresholds)
        self.assertAlmostEqual(self.p.threshold_risk, worst, places=3)

    def test_one_sided_record_is_carried_into_the_issue_prediction(self):
        flagged = [i for i in self.p.issue_predictions if i.one_sided_record]
        self.assertTrue(flagged)
        self.assertTrue(any(i.caveats for i in flagged))

    def test_unargued_threshold_is_a_small_risk_not_even_money(self):
        consumer = next((i for i in self.p.issue_predictions
                         if i.issue_id == "consumer_status"), None)
        if consumer is not None:
            self.assertGreater(consumer.probability, 0.6)


# --------------------------------------------------------------------------
class TestMeritsPath(unittest.TestCase):
    def setUp(self):
        self.p = full_state(UNCONTESTED_CASE)["prediction"]

    def test_uncontested_complaint_is_predicted_for_the_complainant(self):
        self.assertEqual(self.p.favours, "complainant")
        self.assertIn(self.p.disposition, {ALLOWED, PARTLY_ALLOWED})
        self.assertEqual(self.p.decided_at, "merits")

    def test_posterior_is_exactly_prior_plus_drivers(self):
        """The transparency contract: the score must decompose exactly."""
        total = logit(self.p.prior) + sum(d.logit_contribution for d in self.p.drivers)
        self.assertAlmostEqual(total, self.p.posterior_logit, places=3)

    def test_every_driver_is_named_and_directed(self):
        for d in self.p.drivers:
            self.assertTrue(d.name)
            self.assertIn(d.direction, {"complainant", "opposite_party", "neutral"})
            self.assertTrue(d.explanation)

    def test_merits_balance_is_a_driver(self):
        self.assertIn("merits_balance", [d.name for d in self.p.drivers])

    def test_gaps_push_against_the_complainant(self):
        gap = next((d for d in self.p.drivers if d.name == "evidential_gaps"), None)
        if gap is not None:
            self.assertLess(gap.logit_contribution, 0.0)
            self.assertEqual(gap.direction, "opposite_party")

    def test_every_issue_gets_a_predicted_finding(self):
        amap = full_state(UNCONTESTED_CASE)["argument_map"]
        self.assertEqual(len(self.p.issue_predictions), len(amap.issues))


# --------------------------------------------------------------------------
class TestPrior(unittest.TestCase):
    def test_prior_is_shrunk_not_taken_raw(self):
        state = full_state(UNCONTESTED_CASE)
        p, pb = state["prediction"], state["precedent_bundle"]
        raw = pb.authority_weighted_success_rate
        if raw is not None and raw not in (NEUTRAL_PRIOR,):
            self.assertNotAlmostEqual(p.prior, raw, places=3)
            self.assertLess(abs(p.prior - NEUTRAL_PRIOR), abs(raw - NEUTRAL_PRIOR) + 1e-9)

    def test_prior_basis_is_stated(self):
        self.assertTrue(full_state(UNCONTESTED_CASE)["prediction"].prior_basis)

    def test_ablating_the_prior_gives_a_neutral_start(self):
        p = full_state(UNCONTESTED_CASE, agent=PredictionAgent(use_precedent_prior=False))["prediction"]
        self.assertEqual(p.prior, NEUTRAL_PRIOR)
        self.assertIn("ablation", p.prior_basis)

    def test_no_precedent_bundle_still_predicts_with_a_warning(self):
        p = full_state(UNCONTESTED_CASE, with_precedents=False)["prediction"]
        self.assertEqual(p.prior, NEUTRAL_PRIOR)
        self.assertTrue(any("precedent" in w.lower() for w in p.warnings))

    def test_unverified_precedents_are_listed_when_excluded(self):
        p = full_state(BUILDER_CASE)["prediction"]
        self.assertIsInstance(p.excluded_from_prior, list)


# --------------------------------------------------------------------------
class TestCalibrationAndAbstention(unittest.TestCase):
    def test_confidence_is_scaled_by_evidence_quality(self):
        p = full_state(UNCONTESTED_CASE)["prediction"]
        directional = abs(p.probability_complainant_succeeds - 0.5) * 2
        self.assertLessEqual(p.confidence, directional + 1e-9)

    def test_evidence_components_are_reported(self):
        p = full_state(UNCONTESTED_CASE)["prediction"]
        self.assertEqual(set(p.evidence_components), set(EVIDENCE_WEIGHTS))

    def test_high_abstain_threshold_forces_indeterminate(self):
        p = full_state(UNCONTESTED_CASE, agent=PredictionAgent(abstain_threshold=0.99))["prediction"]
        self.assertTrue(p.abstained)
        self.assertEqual(p.disposition, INDETERMINATE)
        self.assertEqual(p.favours, "unknown")
        self.assertTrue(p.abstention_reasons)

    def test_abstention_still_reports_the_probabilities_it_declined_to_call(self):
        p = full_state(UNCONTESTED_CASE, agent=PredictionAgent(abstain_threshold=0.99))["prediction"]
        self.assertTrue(p.disposition_probabilities)

    def test_zero_threshold_forces_a_prediction(self):
        p = full_state(UNCONTESTED_CASE, agent=PredictionAgent(abstain_threshold=0.0))["prediction"]
        self.assertFalse(p.abstained)
        self.assertNotEqual(p.disposition, INDETERMINATE)

    def test_disposition_probabilities_sum_to_one(self):
        p = full_state(BUILDER_CASE)["prediction"]
        self.assertAlmostEqual(sum(p.disposition_probabilities.values()), 1.0, places=3)

    def test_disposition_is_the_argmax_of_its_probabilities(self):
        p = full_state(UNCONTESTED_CASE)["prediction"]
        self.assertEqual(p.disposition, max(p.disposition_probabilities,
                                            key=p.disposition_probabilities.get))


# --------------------------------------------------------------------------
class TestReliefForecast(unittest.TestCase):
    def setUp(self):
        self.rf = full_state(UNCONTESTED_CASE)["prediction"].relief_forecast

    def test_forecast_is_conditional_not_asserted(self):
        self.assertIn("succeed", self.rf.conditional_on)

    def test_heads_are_grounded_in_both_the_claim_and_the_corpus(self):
        # Nothing may be forecast that the complainant never asked for.
        self.assertNotIn("possession", self.rf.relief_heads)

    def test_claimed_but_unprecedented_heads_are_surfaced_not_dropped(self):
        self.assertIsInstance(self.rf.claimed_heads_without_precedent, list)

    def test_interest_rate_is_grounded_in_named_decisions(self):
        if self.rf.typical_interest_rate_pct is not None:
            self.assertTrue(self.rf.interest_rate_basis)

    def test_note_disclaims_quantum(self):
        self.assertTrue(self.rf.note)

    def test_no_fact_sheet_means_no_forecast(self):
        state = full_state(UNCONTESTED_CASE)
        p = PredictionAgent().predict(argument_map=state["argument_map"], fact_sheet=None)
        self.assertEqual(p.relief_forecast.relief_heads, [])


# --------------------------------------------------------------------------
class TestCounterfactuals(unittest.TestCase):
    def setUp(self):
        self.p = full_state(BUILDER_CASE)["prediction"]

    def test_counterfactuals_are_produced(self):
        self.assertTrue(self.p.counterfactuals)

    def test_clearing_the_threshold_flips_a_threshold_dismissal(self):
        cf = next(c for c in self.p.counterfactuals if "threshold" in c.change)
        self.assertGreater(cf.revised_probability, self.p.probability_complainant_succeeds)
        self.assertTrue(cf.flips_prediction)

    def test_counterfactual_probabilities_are_valid(self):
        for c in self.p.counterfactuals:
            self.assertGreaterEqual(c.revised_probability, 0.0)
            self.assertLessEqual(c.revised_probability, 1.0)
            self.assertTrue(c.revised_disposition)


# --------------------------------------------------------------------------
class TestLLMBoundedness(unittest.TestCase):
    def test_llm_cannot_overturn_a_threshold_dismissal(self):
        """
        A bounded opinion may shade the merits and may increase uncertainty to
        the point of abstention - it may NEVER convert a case with a live
        dispositive ground into a predicted complainant win.

        The abstention case is the model behaving correctly: an LLM that
        contradicts the grounded evidence should make the system less sure, not
        hand it a new answer.
        """
        rules = full_state(BUILDER_CASE)["prediction"]
        biased = full_state(BUILDER_CASE, agent=PredictionAgent(llm=StubLLM(1.0)))["prediction"]
        self.assertTrue(rules.dispositive_grounds)
        self.assertNotEqual(biased.favours, "complainant")
        self.assertIn(biased.disposition, {DISMISSED, INDETERMINATE})

    def test_llm_bias_moves_probability_only_marginally(self):
        """The clamp must keep the LLM's influence small next to the evidence."""
        rules = full_state(BUILDER_CASE)["prediction"]
        biased = full_state(BUILDER_CASE, agent=PredictionAgent(llm=StubLLM(1.0)))["prediction"]
        self.assertGreater(biased.probability_complainant_succeeds,
                           rules.probability_complainant_succeeds)
        self.assertLess(biased.probability_complainant_succeeds
                        - rules.probability_complainant_succeeds, 0.20)

    def test_llm_contribution_is_reported_as_its_own_driver(self):
        p = full_state(UNCONTESTED_CASE, agent=PredictionAgent(llm=StubLLM(-1.0)))["prediction"]
        names = [d.name for d in p.drivers]
        self.assertIn("llm_merits_assessment", names)
        self.assertEqual(p.ranked_by, "rules+llm")

    def test_llm_shift_is_clamped(self):
        from agents.prediction import LLM_MAX_LOGIT_SHIFT
        p = full_state(UNCONTESTED_CASE, agent=PredictionAgent(llm=StubLLM(5.0)))["prediction"]
        d = next(d for d in p.drivers if d.name == "llm_merits_assessment")
        self.assertLessEqual(abs(d.logit_contribution), LLM_MAX_LOGIT_SHIFT + 1e-9)

    def test_llm_failure_falls_back_to_rules(self):
        p = full_state(UNCONTESTED_CASE, agent=PredictionAgent(llm=StubLLM(fail=True)))["prediction"]
        self.assertEqual(p.ranked_by, "rules")
        self.assertTrue(any("LLM" in w for w in p.warnings))
        self.assertNotIn("llm_merits_assessment", [d.name for d in p.drivers])


# --------------------------------------------------------------------------
class TestLeakageControl(unittest.TestCase):
    def test_prediction_never_reads_the_case_text(self):
        """Agent 5 must work from bundles alone - state['case_text'] is not an input."""
        state = full_state(UNCONTESTED_CASE)
        p = PredictionAgent().predict(
            argument_map=state["argument_map"], precedent_bundle=state["precedent_bundle"],
            statute_bundle=state["statute_bundle"], fact_sheet=state["fact_sheet"])
        self.assertEqual(p.disposition, state["prediction"].disposition)

    def test_unstripped_outcome_is_flagged_as_leakage_suspect(self):
        state = {"case_text": UNCONTESTED_CASE, "query": ""}
        state = FactExtractionAgent(strip_outcome=False).run(state)
        state = StatuteRetrievalAgent(conn=fresh_db()).run(state)
        state = ArgumentAnalysisAgent().run(state)
        p = PredictionAgent().run(state)["prediction"]
        if not state["fact_sheet"].outcome_removed:
            self.assertTrue(p.leakage_suspect)
            self.assertTrue(any("backtest" in w for w in p.warnings))


# --------------------------------------------------------------------------
class TestDownstreamContracts(unittest.TestCase):
    def setUp(self):
        self.p = full_state(BUILDER_CASE)["prediction"]

    def test_to_dict_is_serialisable(self):
        import json
        json.dumps(self.p.to_dict(), default=str)

    def test_for_explanation_is_irac_ready(self):
        d = self.p.for_explanation()
        for key in ("disposition", "confidence", "issues", "reasons",
                    "must_concede", "relief", "counterfactuals"):
            self.assertIn(key, d)

    def test_for_verification_lists_what_was_relied_on(self):
        d = self.p.for_verification()
        self.assertIn("authorities_relied_on", d)
        self.assertIn("excluded_from_prior", d)

    def test_state_carries_the_explanation_payload(self):
        state = full_state(BUILDER_CASE)
        self.assertIn("prediction_summary", state)
        self.assertEqual(state["prediction_summary"]["disposition"],
                         state["prediction"].disposition)

    def test_summary_renders(self):
        self.assertIn("PREDICTED DISPOSITION", self.p.summary())

    def test_abstained_summary_renders(self):
        p = full_state(BUILDER_CASE, agent=PredictionAgent(abstain_threshold=0.99))["prediction"]
        self.assertIn("INDETERMINATE", p.summary())


# --------------------------------------------------------------------------
class TestBacktestScoring(unittest.TestCase):
    def test_direction_and_exact_are_scored_separately(self):
        p = Prediction(disposition=PARTLY_ALLOWED, favours="complainant",
                       probability_complainant_succeeds=0.8, confidence=0.6)
        r = score_against_actual(p, ALLOWED)
        self.assertTrue(r["direction_correct"])
        self.assertFalse(r["exact_correct"])

    def test_brier_penalises_confident_errors(self):
        confident_wrong = score_against_actual(
            Prediction(disposition=ALLOWED, favours="complainant",
                       probability_complainant_succeeds=0.95), DISMISSED)
        hedged_wrong = score_against_actual(
            Prediction(disposition=ALLOWED, favours="complainant",
                       probability_complainant_succeeds=0.55), DISMISSED)
        self.assertGreater(confident_wrong["brier"], hedged_wrong["brier"])

    def test_abstention_is_not_scored_as_a_miss(self):
        r = score_against_actual(Prediction(abstained=True), DISMISSED)
        self.assertIsNone(r["direction_correct"])
        self.assertIsNone(r["exact_correct"])
        self.assertTrue(r["abstained"])

    def test_leakage_flag_is_carried_into_the_result(self):
        r = score_against_actual(
            Prediction(disposition=ALLOWED, favours="complainant", leakage_suspect=True), ALLOWED)
        self.assertTrue(r["leakage_suspect"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
