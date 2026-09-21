"""
Run:  python -m unittest tests.test_argument_analysis -v

The precedent corpus reused here is the SYNTHETIC fixture set from
tests/test_precedent_retrieval.py - not real decisions.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.argument_analysis import (
    ArgumentAnalysisAgent, ArgumentMap, ISSUE_TEMPLATES, STRENGTH_WEIGHTS,
    classify_contention, grounding_score, leaning_from, party_for_label,
)
from agents.fact_extraction import FactExtractionAgent
from agents.precedent_retrieval import PrecedentRetrievalAgent
from agents.statute_retrieval import StatuteRetrievalAgent
from structuring.db import get_conn
from tests.test_precedent_retrieval import fresh_db

SAMPLE = Path("data/ncdrc_judgments/gunjan_aggarwal_257_2019.txt")

BUILDER_CASE = (
    "The complainant booked a residential flat with the opposite party on 29.12.2012 with a "
    "booking amount of Rs.7,50,000/-. An agreement was executed between the parties on 23.4.2013. "
    "The possession was to be delivered within a period of 48 months. The construction of the "
    "apartment is not complete till date on 15.08.2021, though the complainant has already paid "
    "more than Rs.70 lakhs to the developer. The opposite party failed to deliver possession of "
    "the flat on 10.05.2017 as promised under the agreement. The complainant is seeking refund of "
    "the amount paid along with interest and compensation of Rs.5,00,000. The complaint has been "
    "resisted by the OP which has taken a preliminary objection that the complaint is barred by "
    "limitation. The OP has also contended that this Commission lacks pecuniary jurisdiction. "
    "Dated : 15 Aug 2021"
)

UNCONTESTED_CASE = (
    "I purchased a Samsung refrigerator on 12 March 2024 by paying Rs.45,000/- online. "
    "The compressor became defective within two months and stopped cooling on 20 May 2024. "
    "The seller failed to repair or replace the defective product despite repeated requests. "
    "I am seeking refund of the price along with compensation of Rs.10,000. Dated : 01 Aug 2024"
)


def full_state(case_text, query="", conn=None):
    """Run Agents 1 -> 2 -> 3 -> 4 the way the pipeline does."""
    conn = conn or fresh_db()
    state = {"case_text": case_text, "query": query}
    state = FactExtractionAgent().run(state)
    state = StatuteRetrievalAgent(conn=conn).run(state)
    state = PrecedentRetrievalAgent(conn=conn).run(state)
    return ArgumentAnalysisAgent().run(state)


# --------------------------------------------------------------------------
class TestHelpers(unittest.TestCase):
    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(STRENGTH_WEIGHTS.values()), 1.0, places=9)

    def test_issue_templates_are_well_formed(self):
        for iid, meta in ISSUE_TEMPLATES.items():
            self.assertIn(meta["kind"], ("threshold", "merits", "relief"))
            self.assertIn(meta["burden"], ("complainant", "opposite_party"))
            self.assertTrue(meta["question"] and meta["sections"] is not None)

    def test_contention_classification(self):
        self.assertIn("limitation", classify_contention("The complaint is barred by limitation."))
        self.assertIn("deficiency", classify_contention("The OP failed to deliver possession."))
        self.assertIn("relief", classify_contention("The complainant seeks refund with interest."))
        self.assertEqual(classify_contention("The weather was pleasant that day."), [])

    def test_party_for_label(self):
        self.assertEqual(party_for_label("defence"), "opposite_party")
        self.assertEqual(party_for_label("claim"), "complainant")

    def test_grounding_saturates(self):
        self.assertEqual(grounding_score(0), 0.0)
        self.assertEqual(grounding_score(3), 1.0)
        self.assertEqual(grounding_score(9), 1.0)

    def test_leaning_thresholds(self):
        self.assertEqual(leaning_from(0.0, 0.0)[0], "unsupported")
        self.assertEqual(leaning_from(0.50, 0.48)[0], "contested")
        self.assertEqual(leaning_from(0.80, 0.20)[0], "complainant")
        self.assertEqual(leaning_from(0.20, 0.80)[0], "opposite_party")


class TestIssueFraming(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state = full_state(BUILDER_CASE, "refund for delayed possession")
        cls.map = cls.state["argument_map"]

    def test_returns_a_map(self):
        self.assertIsInstance(self.map, ArgumentMap)
        self.assertTrue(self.map.issues)

    def test_pleaded_issues_are_framed(self):
        ids = [i.id for i in self.map.issues]
        for expected in ("deficiency", "limitation", "jurisdiction", "relief"):
            self.assertIn(expected, ids)

    def test_unraised_issues_are_not_framed(self):
        """A builder-delay case raises no unfair-trade-practice issue."""
        self.assertNotIn("unfair_trade_practice", [i.id for i in self.map.issues])

    def test_issues_record_what_raised_them(self):
        for i in self.map.issues:
            self.assertTrue(i.raised_by)

    def test_threshold_issues_come_first(self):
        kinds = [i.kind for i in self.map.issues]
        self.assertEqual(kinds, sorted(kinds, key=lambda k: {"threshold": 0, "merits": 1, "relief": 2}[k]))


class TestContentionsAndRebuttal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.map = full_state(BUILDER_CASE)["argument_map"]

    def test_contentions_are_traceable_to_fact_ids(self):
        seen = False
        for i in self.map.issues:
            for side in (i.complainant, i.opposite_party):
                for c in side.contentions:
                    seen = True
                    self.assertTrue(c.fact_id.startswith("F"))
                    self.assertTrue(c.text)
                    self.assertIn(c.party, ("complainant", "opposite_party"))
        self.assertTrue(seen)

    def test_defences_are_attributed_to_the_opposite_party(self):
        lim = next(i for i in self.map.issues if i.id == "limitation")
        self.assertTrue(lim.opposite_party.contentions)

    def test_contested_issue_marks_both_sides_rebutted(self):
        lim = next(i for i in self.map.issues if i.id == "limitation")
        if lim.complainant.contentions and lim.opposite_party.contentions:
            self.assertTrue(all(c.rebutted for c in lim.opposite_party.contentions))

    def test_factual_averments_carry_the_merits_case(self):
        """The case on deficiency is made by the facts, not by the prayer for relief."""
        m = full_state(UNCONTESTED_CASE)["argument_map"]
        merits = [i for i in m.issues if i.kind == "merits"]
        self.assertTrue(merits)
        self.assertTrue(any(c.kind == "factual_assertion"
                            for i in merits for c in i.complainant.contentions))

    def test_unrebutted_contentions_are_reported(self):
        uncontested = full_state(UNCONTESTED_CASE)["argument_map"]
        self.assertTrue(uncontested.unrebutted_by_party["complainant"])
        self.assertEqual(uncontested.unrebutted_by_party["opposite_party"], [])


class TestVoiceAttribution(unittest.TestCase):
    """The court is not a party, and on a decided judgment its findings are the outcome."""

    @classmethod
    def setUpClass(cls):
        if not SAMPLE.exists():
            raise unittest.SkipTest("sample judgment not present")
        cls.map = full_state(SAMPLE.read_text(encoding="utf-8"))["argument_map"]

    def test_tribunal_findings_are_excluded(self):
        self.assertTrue(self.map.excluded_tribunal_findings)
        self.assertTrue(any("tribunal's own voice" in w for w in self.map.warnings))

    def test_no_tribunal_finding_became_a_contention(self):
        excluded_ids = {e.split(":")[0] for e in self.map.excluded_tribunal_findings}
        for i in self.map.issues:
            for side in (i.complainant, i.opposite_party):
                for c in side.contentions:
                    self.assertNotIn(c.fact_id, excluded_ids)

    def test_reported_speech_stays_with_the_party(self):
        """'The objection taken by the OP is that this Commission lacks jurisdiction'
        names the Commission but is the OP's plea - it must not be excluded."""
        from agents.argument_analysis import OBJECTION_CUE, REPORTED_SPEECH, TRIBUNAL_VOICE
        s = ("The first preliminary objection taken by the OP is that the complaint is barred by "
             "limitation and the second preliminary objection as taken by the developer is that "
             "this Commission does not have pecuniary jurisdiction.")
        self.assertTrue(TRIBUNAL_VOICE.search(s))
        self.assertTrue(REPORTED_SPEECH.search(s))
        self.assertTrue(OBJECTION_CUE.search(s))
        lim = next(i for i in self.map.issues if i.id == "limitation")
        self.assertTrue(lim.opposite_party.contentions,
                        "the OP's pleaded limitation objection must survive voice filtering")

    def test_a_claim_mentioning_the_forum_is_not_tribunal_voice(self):
        """'The complainants are before this Commission seeking refund' is the claim."""
        from agents.argument_analysis import TRIBUNAL_VOICE
        self.assertFalse(TRIBUNAL_VOICE.search(
            "The complainants are therefore, before this Commission seeking refund of the amount."))
        self.assertTrue(TRIBUNAL_VOICE.search(
            "Therefore, this Commission has the requisite pecuniary jurisdiction."))

    def test_one_sided_record_is_flagged(self):
        flagged = [i for i in self.map.issues if i.one_sided_record]
        for i in flagged:
            self.assertTrue(any("one-sided pleading record" in n for n in i.notes))
        for r in self.map.dispositive_risks:
            self.assertIn("one_sided_record", r)

    def test_unattributable_points_are_not_handed_to_a_party(self):
        for e in self.map.unattributed_contentions:
            self.assertTrue(e)


class TestSupportAttachment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.map = full_state(BUILDER_CASE)["argument_map"]

    def test_statutes_attach_to_the_side_they_help(self):
        lim = next(i for i in self.map.issues if i.id == "limitation")
        self.assertTrue(any(s.section in ("69", "24A") for s in lim.opposite_party.statutory_support))
        def_issue = next(i for i in self.map.issues if i.id == "deficiency")
        self.assertTrue(any(s.section in ("2(11)", "2(1)(g)")
                            for s in def_issue.complainant.statutory_support))

    def test_precedents_attach_by_their_outcome(self):
        for i in self.map.issues:
            for p in i.complainant.precedent_support:
                self.assertEqual(p.favours, "complainant")
            for p in i.opposite_party.precedent_support:
                self.assertEqual(p.favours, "opposite_party")

    def test_threshold_issue_only_takes_precedents_with_a_holding_on_it(self):
        lim = next(i for i in self.map.issues if i.id == "limitation")
        for side in (lim.complainant, lim.opposite_party):
            for p in side.precedent_support:
                self.assertTrue(p.holding)

    def test_precedent_cap_per_issue(self):
        agent = ArgumentAnalysisAgent(max_precedents_per_issue=1)
        state = full_state(BUILDER_CASE)
        m = agent.analyse(state["fact_sheet"], state["statute_bundle"], state["precedent_bundle"])
        for i in m.issues:
            self.assertLessEqual(
                len(i.complainant.precedent_support) + len(i.opposite_party.precedent_support), 1)


class TestVerificationGate(unittest.TestCase):
    def test_unverified_support_is_shown_but_not_counted(self):
        state = full_state(BUILDER_CASE)
        # make every precedent inadmissible and re-analyse
        for m in state["precedent_bundle"].matches:
            m.verified = False
            m.admissible_as_authority = False
        m2 = ArgumentAnalysisAgent().analyse(
            state["fact_sheet"], state["statute_bundle"], state["precedent_bundle"])
        attached = [p for i in m2.issues for side in (i.complainant, i.opposite_party)
                    for p in side.precedent_support]
        self.assertTrue(attached, "inadmissible precedents should still be shown")
        self.assertTrue(all(not p.counts_toward_strength for p in attached))
        self.assertTrue(m2.excluded_support)

    def test_excluded_support_carries_a_reason(self):
        state = full_state(BUILDER_CASE)
        for m in state["precedent_bundle"].matches:
            m.verified = False
            m.admissible_as_authority = False
        m2 = ArgumentAnalysisAgent().analyse(
            state["fact_sheet"], state["statute_bundle"], state["precedent_bundle"])
        for entry in m2.excluded_support:
            self.assertIn(" - ", entry)

    def test_ablation_flag_can_count_unverified(self):
        state = full_state(BUILDER_CASE)
        for m in state["precedent_bundle"].matches:
            m.verified = False
            m.admissible_as_authority = False
        strict = ArgumentAnalysisAgent().analyse(
            state["fact_sheet"], state["statute_bundle"], state["precedent_bundle"])
        loose = ArgumentAnalysisAgent(count_unverified=True).analyse(
            state["fact_sheet"], state["statute_bundle"], state["precedent_bundle"])
        strict_p = sum(i.complainant.strength_components["precedential"] for i in strict.issues)
        loose_p = sum(i.complainant.strength_components["precedential"] for i in loose.issues)
        self.assertGreater(loose_p, strict_p)


class TestScoringAndAggregation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.map = full_state(BUILDER_CASE)["argument_map"]

    def test_every_component_reported(self):
        for i in self.map.issues:
            for side in (i.complainant, i.opposite_party):
                self.assertGreaterEqual(set(side.strength_components), set(STRENGTH_WEIGHTS))
                self.assertGreaterEqual(side.strength, 0.0)
                self.assertLessEqual(side.strength, 1.0)

    def test_a_side_that_pleaded_nothing_scores_nothing(self):
        for i in self.map.issues:
            for side in (i.complainant, i.opposite_party):
                if not side.contentions:
                    self.assertEqual(side.strength, 0.0)

    def test_merits_leans_to_the_complainant_on_an_unanswered_delay(self):
        m = full_state(UNCONTESTED_CASE)["argument_map"]
        self.assertGreater(m.merits_balance, 0.0)

    def test_balance_is_bounded_and_discounted_by_threshold_risk(self):
        self.assertGreaterEqual(self.map.argument_balance, -1.0)
        self.assertLessEqual(self.map.argument_balance, 1.0)
        self.assertLessEqual(abs(self.map.argument_balance), abs(self.map.merits_balance) + 1e-9)

    def test_threshold_risk_is_separate_from_merits(self):
        self.assertGreaterEqual(self.map.threshold_risk, 0.0)
        self.assertLessEqual(self.map.threshold_risk, 1.0)

    def test_dispositive_threshold_is_flagged_not_averaged_away(self):
        """A strong limitation defence must surface even when the merits are strong."""
        state = full_state(BUILDER_CASE)
        amap = state["argument_map"]
        lim = next(i for i in amap.issues if i.id == "limitation")
        if lim.leaning == "opposite_party":
            self.assertTrue(lim.dispositive)
            self.assertIn("limitation", [r["issue_id"] for r in amap.dispositive_risks])

    def test_agent_2_findings_surface_as_notes(self):
        lim = next(i for i in self.map.issues if i.id == "limitation")
        self.assertTrue(any("Agent 2 limitation finding" in n for n in lim.notes))

    def test_evidential_gaps_are_collected(self):
        self.assertIsInstance(self.map.evidential_gaps, list)
        for g in self.map.evidential_gaps:
            self.assertTrue(g.startswith("["))


class TestPredictionHandoff(unittest.TestCase):
    def test_feature_shape(self):
        state = full_state(BUILDER_CASE)
        feats = state["argument_features"]
        for k in ("argument_balance", "merits_balance", "threshold_risk",
                  "dispositive_risks", "issues", "evidential_gap_count"):
            self.assertIn(k, feats)
        for i in feats["issues"]:
            self.assertIn(i["leaning"], ("complainant", "opposite_party", "contested", "unsupported"))

    def test_map_serialises(self):
        m = full_state(UNCONTESTED_CASE)["argument_map"]
        d = m.to_dict()
        self.assertIn("issues", d)
        self.assertTrue(d["issues"][0]["question"])


class TestDegradedInputs(unittest.TestCase):
    def test_no_fact_sheet(self):
        m = ArgumentAnalysisAgent().analyse()
        self.assertEqual(m.issues, [])
        self.assertTrue(any("No fact sheet" in w for w in m.warnings))

    def test_fact_sheet_only(self):
        sheet = FactExtractionAgent().extract(BUILDER_CASE)
        m = ArgumentAnalysisAgent().analyse(fact_sheet=sheet)
        self.assertTrue(m.issues)
        self.assertTrue(any("No statute bundle" in w for w in m.warnings))
        self.assertTrue(any("No precedent bundle" in w for w in m.warnings))
        self.assertEqual(m.inputs_used, ["fact_sheet"])

    def test_without_precedents(self):
        conn = get_conn(":memory:")
        state = {"case_text": BUILDER_CASE, "query": ""}
        state = FactExtractionAgent().run(state)
        state = StatuteRetrievalAgent(conn=conn).run(state)
        m = ArgumentAnalysisAgent().run(state)["argument_map"]
        self.assertTrue(m.issues)
        for i in m.issues:
            self.assertEqual(i.complainant.precedent_support, [])

    def test_thin_case_file(self):
        sheet = FactExtractionAgent().extract(
            "The service was deficient and the opposite party failed to respond to my requests.")
        m = ArgumentAnalysisAgent().analyse(fact_sheet=sheet)
        self.assertIsInstance(m, ArgumentMap)


class TestPipelineContract(unittest.TestCase):
    def test_full_chain_on_the_real_judgment(self):
        if not SAMPLE.exists():
            self.skipTest("sample judgment not present")
        state = full_state(SAMPLE.read_text(encoding="utf-8"), "refund delayed possession")
        for k in ("fact_sheet", "statute_bundle", "precedent_bundle",
                  "argument_map", "argument_features"):
            self.assertIn(k, state)
        amap = state["argument_map"]
        self.assertIn("deficiency", [i.id for i in amap.issues])
        self.assertEqual(amap.inputs_used, ["fact_sheet", "statute_bundle", "precedent_bundle"])

    def test_no_outcome_text_reaches_the_argument_map(self):
        """Agent 4 reads stripped facts only - the operative order must not appear."""
        if not SAMPLE.exists():
            self.skipTest("sample judgment not present")
        state = full_state(SAMPLE.read_text(encoding="utf-8"))
        blob = " ".join(c.text for i in state["argument_map"].issues
                        for side in (i.complainant, i.opposite_party) for c in side.contentions)
        for leaked in ("70,33,103", "9% per annum", "within six months from today"):
            self.assertNotIn(leaked, blob)


class _GoodLLM:
    def complete_json(self, system, user):
        return [{"id": 0, "issues": ["relief"]}]


class _BadLLM:
    def complete_json(self, system, user):
        raise RuntimeError("no network")


class _HallucinatingLLM:
    def complete_json(self, system, user):
        return [{"id": 0, "issues": ["issue_that_does_not_exist"]},
                {"id": 999, "issues": ["relief"]}]


class TestLLMMapping(unittest.TestCase):
    def setUp(self):
        s = full_state(BUILDER_CASE)
        self.args = (s["fact_sheet"], s["statute_bundle"], s["precedent_bundle"])

    def test_llm_can_reassign(self):
        m = ArgumentAnalysisAgent(llm=_GoodLLM()).analyse(*self.args)
        self.assertTrue(m.issues)

    def test_llm_failure_falls_back_to_rules(self):
        m = ArgumentAnalysisAgent(llm=_BadLLM()).analyse(*self.args)
        self.assertTrue(m.issues)
        self.assertTrue(any("LLM argument mapping unavailable" in w for w in m.warnings))

    def test_llm_cannot_invent_an_issue_or_a_contention(self):
        m = ArgumentAnalysisAgent(llm=_HallucinatingLLM()).analyse(*self.args)
        self.assertTrue(all(i.id in ISSUE_TEMPLATES for i in m.issues))
        sheet = self.args[0]
        base_ids = {f.id for f in sheet.claims + sheet.defences
                    + sheet.procedural_points + sheet.facts}
        for i in m.issues:
            for side in (i.complainant, i.opposite_party):
                for c in side.contentions:
                    self.assertIn(c.fact_id, base_ids)


if __name__ == "__main__":
    unittest.main()
