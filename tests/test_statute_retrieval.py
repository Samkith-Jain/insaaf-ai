"""
Run:  python -m unittest tests.test_statute_retrieval -v
"""
import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.fact_extraction import FactExtractionAgent
from agents.precedent_retrieval import PrecedentRetrievalAgent
from agents.statute_retrieval import (
    SCORING_WEIGHTS, StatuteBundle, StatuteRetrievalAgent,
    build_statute_query, cause_of_action_date, fact_corpus, match_elements,
)
from statutes.cpa_corpus import (
    ALL_PROVISIONS, CPA_1986, CPA_2019, CPA_2019_COMMENCEMENT,
    UNMODELLED_EQUIVALENTS, act_in_force, equivalent_is_modelled, forum_for_value,
    get_provision, pecuniary_regime, provision_text,
)
from structuring.db import get_conn
from verification.citeverify import verify_statute

SAMPLE = Path("data/ncdrc_judgments/gunjan_aggarwal_257_2019.txt")

# A builder-delay case (the archetypal fact pattern for this corpus), written
# as a free-text case file so the test does not depend on a shipped judgment.
BUILDER_CASE = (
    "The complainant booked a residential flat with the opposite party on 29.12.2012 with a "
    "booking amount of Rs.7,50,000/-. An agreement was executed between the parties on 23.4.2013. "
    "The possession was to be delivered within a period of 48 months. The construction of the "
    "apartment is not complete till date on 15.08.2021, though the complainant has already paid "
    "more than Rs.70 lakhs to the developer. The opposite party failed to deliver possession of "
    "the flat on 10.05.2017 as promised under the agreement. The complainant is seeking refund of "
    "the amount paid along with interest and compensation of Rs.5,00,000. The OP has resisted the "
    "complaint contending that it is barred by limitation and that this Commission lacks "
    "pecuniary jurisdiction. Dated : 15 Aug 2021"
)

GOODS_CASE = (
    "I purchased a Samsung refrigerator on 12 March 2024 by paying Rs.45,000/- online. "
    "The compressor became defective within two months and stopped cooling on 20 May 2024. "
    "The seller failed to repair or replace the defective product despite repeated requests. "
    "The product was not as described in the advertisement. "
    "I am seeking refund of the price along with compensation of Rs.10,000. Dated : 01 Aug 2024"
)


def sheet_for(text, query=""):
    return FactExtractionAgent().extract(text, query)


# --------------------------------------------------------------------------
class TestCorpus(unittest.TestCase):
    def test_every_provision_is_well_formed(self):
        for p in ALL_PROVISIONS:
            self.assertTrue(p["section"] and p["act"] and p["title"] and p["summary"])
            self.assertTrue(p["elements"], f"{p['section']} has no elements")
            self.assertIn(p["favours"], ("complainant", "opposite_party", "neutral"))
            self.assertFalse(p["verbatim"], "no entry may claim to be gazette text")

    def test_provision_lookup_and_text(self):
        p = get_provision("2(11)", CPA_2019)
        self.assertEqual(p["title"], "Deficiency")
        self.assertIn("deficiency", provision_text(p).lower())
        self.assertIsNone(get_provision("2(11)", "Some Other Act, 1999"))

    def test_equivalents_are_symmetric(self):
        for p in ALL_PROVISIONS:
            eq = p.get("equivalent")
            if not eq:
                continue
            other = get_provision(*eq)
            if other is None:
                # allowed only if declared as outside the modelled 1986 subset
                self.assertIn(tuple(eq), UNMODELLED_EQUIVALENTS,
                              f"{p['section']} points at an undeclared missing equivalent {eq}")
                self.assertFalse(equivalent_is_modelled(p))
                continue
            self.assertTrue(equivalent_is_modelled(p))
            if other.get("equivalent"):
                self.assertEqual(other["equivalent"][0], p["section"])

    def test_chapter_entry_is_flagged_not_cited(self):
        ch = get_provision("Chapter VI (ss. 82-87)", CPA_2019)
        self.assertFalse(ch["numbering_verified"])

    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(SCORING_WEIGHTS.values()), 1.0, places=9)


class TestTemporalApplicability(unittest.TestCase):
    def test_pre_commencement_cause_uses_1986_act(self):
        act, reason = act_in_force(date(2016, 5, 1))
        self.assertEqual(act, CPA_1986)
        self.assertIn("precedes", reason)

    def test_post_commencement_cause_uses_2019_act(self):
        act, _ = act_in_force(date(2021, 1, 1))
        self.assertEqual(act, CPA_2019)

    def test_commencement_date_itself_is_2019_act(self):
        self.assertEqual(act_in_force(CPA_2019_COMMENCEMENT)[0], CPA_2019)

    def test_unknown_date_defaults_and_says_so(self):
        act, reason = act_in_force(None)
        self.assertEqual(act, CPA_2019)
        self.assertIn("defaulting", reason)

    def test_agent_selects_the_1986_act_for_an_old_case(self):
        old = BUILDER_CASE.replace("Dated : 15 Aug 2021", "Dated : 10 Aug 2016")
        b = StatuteRetrievalAgent(conn=get_conn(":memory:")).retrieve(fact_sheet=sheet_for(old))
        self.assertEqual(b.governing_act, CPA_1986)
        self.assertTrue(all(m.act == CPA_1986 for m in b.matches))
        self.assertEqual(b.limitation.provision, "24A")


class TestPecuniaryBands(unittest.TestCase):
    def test_bands_shift_with_the_rules_in_force(self):
        # Rs.60 lakh: District under the as-enacted bands, State under the 2021 bands
        self.assertEqual(forum_for_value(6_000_000, date(2021, 1, 1))[0], "District Commission")
        self.assertEqual(forum_for_value(6_000_000, date(2023, 1, 1))[0], "State Commission")

    def test_large_claim_reaches_the_national_commission(self):
        self.assertEqual(forum_for_value(250_000_000, date(2023, 1, 1))[0], "National Commission")

    def test_regime_selection_by_date(self):
        self.assertEqual(pecuniary_regime(date(2015, 1, 1))["act"], CPA_1986)
        self.assertEqual(pecuniary_regime(None)["regime"],
                         "CPA 2019 (as revised by the 2021 jurisdiction rules)")


class TestElementMatching(unittest.TestCase):
    def setUp(self):
        self.corpus = fact_corpus(sheet_for(BUILDER_CASE))

    def test_elements_are_grounded_in_fact_ids(self):
        findings = match_elements(get_provision("2(11)", CPA_2019), self.corpus)
        self.assertTrue(findings)
        satisfied = [f for f in findings if f.satisfied]
        self.assertTrue(satisfied)
        for f in satisfied:
            self.assertTrue(f.fact_ids)
            self.assertTrue(f.evidence)
            self.assertTrue(all(fid.startswith(("F", "Q")) for fid in f.fact_ids))

    def test_unsatisfied_elements_are_reported_not_hidden(self):
        # a builder-delay case establishes no defect in *goods*
        findings = match_elements(get_provision("2(10)", CPA_2019), self.corpus)
        self.assertTrue(any(not f.satisfied for f in findings))

    def test_goods_case_satisfies_the_defect_elements(self):
        corpus = fact_corpus(sheet_for(GOODS_CASE))
        findings = match_elements(get_provision("2(10)", CPA_2019), corpus)
        self.assertTrue(all(f.satisfied for f in findings))


class TestQueryConstruction(unittest.TestCase):
    def test_query_leads_with_statutory_concepts(self):
        q = build_statute_query(sheet_for(BUILDER_CASE), "refund of flat")
        self.assertTrue(q)
        self.assertIn("deficiency", q)
        self.assertEqual(len(q.split()), len(set(q.split())))

    def test_empty_inputs_give_an_empty_query(self):
        self.assertEqual(build_statute_query(None, ""), "")


class TestRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn = get_conn(":memory:")
        cls.sheet = sheet_for(BUILDER_CASE)
        cls.bundle = StatuteRetrievalAgent(conn=cls.conn).retrieve(
            fact_sheet=cls.sheet, user_query="refund for delayed possession")

    def test_returns_ranked_matches(self):
        self.assertIsInstance(self.bundle, StatuteBundle)
        self.assertTrue(self.bundle.matches)
        scores = [m.final_score for m in self.bundle.matches]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_deficiency_is_retrieved_for_a_builder_delay_case(self):
        self.assertIn("2(11)", [m.section for m in self.bundle.matches])

    def test_cause_of_action_outranks_the_limitation_defence(self):
        """A builder-delay case is founded on s.2(11); s.69 is the defence to it."""
        sections = [m.section for m in self.bundle.matches]
        self.assertIn("69", sections)
        self.assertLess(sections.index("2(11)"), sections.index("69"))

    def test_roles_are_reported(self):
        by_section = {m.section: m for m in self.bundle.matches}
        self.assertEqual(by_section["2(11)"].role, "cause_of_action")
        self.assertEqual(by_section["69"].role, "limitation")

    def test_defect_in_goods_does_not_outrank_deficiency_here(self):
        sections = [m.section for m in self.bundle.matches]
        if "2(10)" in sections:
            self.assertLess(sections.index("2(11)"), sections.index("2(10)"))

    def test_every_component_and_provenance_reported(self):
        for m in self.bundle.matches:
            self.assertGreaterEqual(set(m.score_components), set(SCORING_WEIGHTS))
            self.assertTrue(m.provenance)
            self.assertTrue(m.why_relevant)

    def test_sides_are_separated_for_argument_analysis(self):
        self.assertTrue(self.bundle.complainant_provisions)
        # the OP pleaded limitation, so s.69 should be on its side of the sheet
        self.assertTrue(any("69" in c for c in self.bundle.opposite_party_provisions))

    def test_matches_are_verified_through_citeverify(self):
        for m in self.bundle.matches:
            self.assertTrue(m.verification_reason)
        self.assertEqual(self.bundle.verified_count + self.bundle.unverified_count,
                         len(self.bundle.matches))
        self.assertGreater(self.bundle.verified_count, 0)

    def test_summaries_never_claim_to_be_statutory_text(self):
        self.assertFalse(self.bundle.verbatim_text_available)
        for m in self.bundle.matches:
            self.assertFalse(m.verbatim_text_available)

    def test_top_k_and_min_score_respected(self):
        b = StatuteRetrievalAgent(conn=self.conn, top_k=2).retrieve(fact_sheet=self.sheet)
        self.assertLessEqual(len(b.matches), 2)
        b2 = StatuteRetrievalAgent(conn=self.conn, min_score=0.99).retrieve(fact_sheet=self.sheet)
        self.assertEqual(b2.matches, [])
        self.assertTrue(any("min_score" in w for w in b2.warnings))

    def test_other_act_shown_flagged_when_requested(self):
        b = StatuteRetrievalAgent(conn=self.conn, top_k=20, include_other_act=True).retrieve(
            fact_sheet=self.sheet)
        off = [m for m in b.matches if not m.in_force_for_this_case]
        for m in off:
            self.assertTrue(any("NOT in force" in w for w in m.why_relevant))


class TestStatutoryTests(unittest.TestCase):
    def test_jurisdiction_computed_from_the_amount_in_issue(self):
        b = StatuteRetrievalAgent(conn=get_conn(":memory:")).retrieve(fact_sheet=sheet_for(BUILDER_CASE))
        j = b.jurisdiction
        self.assertTrue(j.determinate)
        self.assertGreaterEqual(j.amount_in_issue_inr, 7_000_000)
        self.assertIn(j.forum, ("State Commission", "National Commission", "District Commission"))
        self.assertTrue(j.inputs)

    def test_jurisdiction_indeterminate_without_an_amount(self):
        b = StatuteRetrievalAgent(conn=get_conn(":memory:")).retrieve(
            fact_sheet=sheet_for("The service was deficient and the work was never completed till date. "
                                 "The opposite party failed to respond to repeated requests."))
        self.assertFalse(b.jurisdiction.determinate)
        self.assertTrue(b.jurisdiction.note)

    def test_limitation_within_time(self):
        b = StatuteRetrievalAgent(conn=get_conn(":memory:")).retrieve(fact_sheet=sheet_for(GOODS_CASE))
        self.assertEqual(b.limitation.status, "within_time")

    def test_continuing_wrong_is_not_declared_time_barred(self):
        b = StatuteRetrievalAgent(conn=get_conn(":memory:")).retrieve(fact_sheet=sheet_for(BUILDER_CASE))
        lim = b.limitation
        self.assertTrue(lim.continuing_cause_indicated)
        self.assertNotEqual(lim.status, "prima_facie_barred")

    def test_stale_claim_is_flagged_prima_facie_barred(self):
        stale = ("The complainant purchased a vehicle on 01.01.2010 for Rs.5,00,000. "
                 "The vehicle was found defective and the dealer failed to repair it on 01.02.2010. "
                 "The complainant is seeking refund. Dated : 01 Mar 2020")
        b = StatuteRetrievalAgent(conn=get_conn(":memory:")).retrieve(fact_sheet=sheet_for(stale))
        self.assertEqual(b.limitation.status, "prima_facie_barred")
        self.assertIn("condonation", b.limitation.note)

    def test_limitation_not_guessed_without_dates(self):
        b = StatuteRetrievalAgent(conn=get_conn(":memory:")).retrieve(
            fact_sheet=sheet_for("The service was deficient and the opposite party failed to respond."))
        self.assertEqual(b.limitation.status, "indeterminate")
        self.assertIn("not computed", b.limitation.note)

    def test_cause_of_action_prefers_the_breach_not_the_purchase(self):
        coa, inputs = cause_of_action_date(sheet_for(GOODS_CASE))
        self.assertEqual(coa, "2024-05-20")
        self.assertTrue(inputs)


class TestCorpusSupportSignal(unittest.TestCase):
    def test_citations_in_the_db_lift_a_provision(self):
        conn = get_conn(":memory:")
        conn.execute("INSERT INTO judgments (source_path, full_text) VALUES ('fixture', 'text')")
        for _ in range(5):
            conn.execute("INSERT INTO statute_citations (judgment_id, section, act) VALUES (1, ?, ?)",
                         ("2(11)", CPA_2019))
        conn.commit()
        with_db = StatuteRetrievalAgent(conn=conn).retrieve(fact_sheet=sheet_for(BUILDER_CASE))
        without = StatuteRetrievalAgent(conn=get_conn(":memory:")).retrieve(
            fact_sheet=sheet_for(BUILDER_CASE))
        a = next(m for m in with_db.matches if m.section == "2(11)")
        b = next(m for m in without.matches if m.section == "2(11)")
        self.assertEqual(a.corpus_citations, 5)
        self.assertGreater(a.final_score, b.final_score)
        self.assertIn("cited_by_ingested_judgments", a.provenance)

    def test_runs_without_a_usable_db(self):
        b = StatuteRetrievalAgent(conn=None, db_path="/nonexistent/dir/legal.db").retrieve(
            fact_sheet=sheet_for(GOODS_CASE))
        self.assertTrue(b.matches)


class TestEdgeCases(unittest.TestCase):
    def test_no_facts_is_reported_not_guessed(self):
        b = StatuteRetrievalAgent(conn=get_conn(":memory:")).retrieve(fact_sheet=None, case_text="")
        self.assertEqual(b.matches, [])
        self.assertTrue(b.warnings)

    def test_works_from_raw_text_without_a_fact_sheet(self):
        b = StatuteRetrievalAgent(conn=get_conn(":memory:")).retrieve(case_text=GOODS_CASE)
        self.assertTrue(b.matches)

    def test_unknown_reference_date_warns(self):
        b = StatuteRetrievalAgent(conn=get_conn(":memory:")).retrieve(
            fact_sheet=sheet_for("The seller failed to replace the defective product despite requests."))
        self.assertTrue(any("No reference date" in w for w in b.warnings))


class TestPipelineContract(unittest.TestCase):
    def test_agent_2_output_feeds_agent_3(self):
        """Agent 3 already reads state['statutes']; Agent 2 must fill it in that shape."""
        conn = get_conn(":memory:")
        state = {"case_text": BUILDER_CASE, "query": "refund for delayed possession"}
        state = FactExtractionAgent().run(state)
        state = StatuteRetrievalAgent(conn=conn).run(state)
        for k in ("statute_bundle", "statutes", "statute_query"):
            self.assertIn(k, state)
        self.assertTrue(state["statutes"])
        for s in state["statutes"]:
            self.assertIn("section", s)
            self.assertIn("act", s)
        # and Agent 3 consumes it without modification
        state = PrecedentRetrievalAgent(conn=conn).run(state)
        self.assertIn("precedent_bundle", state)
        self.assertIn("section", state["precedent_bundle"].precedent_query.lower())

    def test_runs_on_the_real_judgment(self):
        if not SAMPLE.exists():
            self.skipTest("sample judgment not present")
        text = SAMPLE.read_text(encoding="utf-8")
        state = StatuteRetrievalAgent(conn=get_conn(":memory:")).run(
            FactExtractionAgent().run({"case_text": text, "query": ""}))
        b = state["statute_bundle"]
        self.assertTrue(b.matches)
        self.assertIn("2(11)", [m.section for m in b.matches])
        self.assertEqual(b.reference_date, "2020-09-28")
        self.assertEqual(b.governing_act, CPA_2019)


class TestCiteVerifyIntegration(unittest.TestCase):
    def test_sub_clauses_now_authenticate(self):
        """Registering the corpus sub-clauses is what makes s.2(11) verifiable."""
        for section in ("2(11)", "2(47)", "69"):
            self.assertTrue(verify_statute(section, CPA_2019).verified, section)
        self.assertTrue(verify_statute("24A", CPA_1986).verified)

    def test_unknown_section_still_fails(self):
        self.assertFalse(verify_statute("999", CPA_2019).verified)

    def test_hand_curated_titles_are_not_overwritten(self):
        from verification.curated_provisions import CPA_2019_SECTIONS
        self.assertEqual(CPA_2019_SECTIONS["2"], "Definitions")


class _GoodLLM:
    def complete_json(self, system, user):
        return [{"id": 0, "applicability": 0.1, "reason": "peripheral"},
                {"id": 1, "applicability": 1.0, "reason": "directly governs"}]


class _BadLLM:
    def complete_json(self, system, user):
        raise RuntimeError("no network")


class _HallucinatingLLM:
    def complete_json(self, system, user):
        return [{"id": 77, "applicability": 1.0, "reason": "invented section"}]


class TestLLMRescoring(unittest.TestCase):
    def test_llm_rescores_the_shortlist(self):
        b = StatuteRetrievalAgent(conn=get_conn(":memory:"), llm=_GoodLLM()).retrieve(
            fact_sheet=sheet_for(BUILDER_CASE))
        self.assertTrue(any("llm_applicability" in m.score_components for m in b.matches))

    def test_llm_failure_falls_back_to_rules(self):
        b = StatuteRetrievalAgent(conn=get_conn(":memory:"), llm=_BadLLM()).retrieve(
            fact_sheet=sheet_for(BUILDER_CASE))
        self.assertTrue(b.matches)
        self.assertTrue(all(m.ranked_by == "rules" for m in b.matches))
        self.assertTrue(any("LLM re-scoring unavailable" in w for w in b.warnings))

    def test_llm_cannot_introduce_a_section(self):
        b = StatuteRetrievalAgent(conn=get_conn(":memory:"), llm=_HallucinatingLLM()).retrieve(
            fact_sheet=sheet_for(BUILDER_CASE))
        known = {(p["section"], p["act"]) for p in ALL_PROVISIONS}
        for m in b.matches:
            self.assertIn((m.section, m.act), known)


if __name__ == "__main__":
    unittest.main()
