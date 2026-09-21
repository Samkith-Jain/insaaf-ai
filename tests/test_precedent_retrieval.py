"""
Run:  python -m unittest tests.test_precedent_retrieval -v

The corpus below is SYNTHETIC. It is written in the shape of NCDRC/State
Commission orders purely as a test fixture - none of these are real
decisions and none are shipped into data/. The one real judgment in the
repo (gunjan_aggarwal_257_2019.txt) is used separately, as the pending case.
"""
import re
import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.fact_extraction import FactExtractionAgent
from agents.precedent_retrieval import (
    PrecedentRetrievalAgent, PrecedentBundle, SCORING_WEIGHTS,
    authority_weight, build_precedent_query, case_signals, detect_forum_extended,
    extract_holdings, extract_outcome, jaccard, recency_score,
)
from structuring.db import get_conn

SAMPLE = Path("data/ncdrc_judgments/gunjan_aggarwal_257_2019.txt")


def _judgment(forum_header, case_no, dated, body, order):
    return f"""{forum_header}
NEW DELHI

{case_no}

1. A COMPLAINANT
...........Complainant(s)
Versus\t
1. AN OPPOSITE PARTY PVT. LTD.
...........Opp.Party(s)

BEFORE:\t
\tHON'BLE MR. JUSTICE A. B. SINGH,PRESIDING MEMBER

Dated : {dated}
ORDER
JUSTICE A. B. SINGH, PRESIDING MEMBER (ORAL)

{body}

{order}
"""


FIXTURES = [
    # 0: real-estate, delayed possession, allowed -> the governing precedent
    _judgment(
        "NATIONAL CONSUMER DISPUTES REDRESSAL COMMISSION", "CONSUMER CASE NO. 111 OF 2018", "12 Mar 2019",
        """          The complainant booked a residential flat with the opposite party in a project on 01.02.2013 with a booking amount of Rs.5,00,000/-.  Thereafter, a buyer's agreement was executed between the parties on 15.3.2013.

2.      It would thus be seen that the possession was to be delivered within a period of 36 months.

3.      The grievance of the complainant is that the construction of the apartment is not complete till date, though he has already made payment of more than Rs.60 lakhs to the developer.  The complainant is therefore, before this Commission seeking refund of the amount paid along with interest and compensation.

4.      The complaint has been resisted by the OP which has taken a preliminary objection that the complaint is barred by limitation.

5.      As far as limitation is concerned, since the OP did not deliver or even offer the possession of the allotted flat, the complainant had a recurrent cause of action to file a consumer complaint.

6.      We are of the considered view that the failure of the developer to deliver possession within the committed period clearly amounts to deficiency in service in terms of Section 2(11) of the Consumer Protection Act, 2019.  The complainant cannot be compelled to wait indefinitely for possession.""",
        """7.      The complaint is allowed and disposed of with the following directions:
(i)     The opposite party shall refund the entire principal amount of Rs.60,00,000/- to the complainant along with interest @ 9% per annum.
(ii)    The opposite party shall also pay Rs.25,000/- as cost of litigation."""),

    # 1: real-estate, delayed possession, dismissed on limitation -> contrary authority
    _judgment(
        "STATE CONSUMER DISPUTES REDRESSAL COMMISSION", "CONSUMER CASE NO. 222 OF 2015", "05 Jun 2016",
        """          The complainant booked a flat on 10.01.2007 and an agreement was executed on 20.2.2007. Possession was to be delivered within a period of 24 months.

2.      The complainant has approached this Commission in 2015 seeking refund of Rs.15,00,000/- paid to the builder along with compensation.

3.      The OP contended that the complaint is barred by limitation as it has been filed more than two years after the cause of action arose.

4.      We are of the opinion that the complainant slept over his rights and the complaint is hopelessly barred by limitation.""",
        """5.      The complaint is dismissed as barred by limitation, with no order as to costs."""),

    # 2: insurance repudiation, partly allowed -> different fact pattern
    _judgment(
        "NATIONAL CONSUMER DISPUTES REDRESSAL COMMISSION", "CONSUMER CASE NO. 333 OF 2017", "19 Nov 2018",
        """          The complainant had taken a mediclaim policy and paid a premium of Rs.24,000/- for a sum assured of Rs.5,00,000/-.

2.      On hospitalisation, the claim was repudiated by the insurer on the ground of a pre-existing disease.

3.      It is well settled that repudiation of a claim on a ground not disclosed in the policy amounts to deficiency in service under the Consumer Protection Act, 2019.""",
        """4.      The complaint is partly allowed. The opposite party is directed to pay Rs.3,00,000/- to the complainant with interest @ 7% per annum."""),

    # 3: defective goods, allowed -> different fact pattern, small amount
    _judgment(
        "DISTRICT CONSUMER DISPUTES REDRESSAL COMMISSION", "CONSUMER CASE NO. 444 OF 2021", "08 Feb 2022",
        """          The complainant purchased a refrigerator on 05.04.2021 by paying Rs.42,000/-. The compressor was found to be defective within two months.

2.      Despite repeated requests the seller failed to repair or replace the product.

3.      We are of the view that supply of a defective product and the refusal to replace it amounts to deficiency in service.""",
        """4.      The complaint is allowed. The opposite party shall replace the product and pay Rs.5,000/- as compensation."""),
]


def seed(conn, texts=None):
    texts = FIXTURES if texts is None else texts
    for t in texts:
        forum = detect_forum_extended(t)
        case_no = re.search(r"(CONSUMER CASE NO\.\s*\d+\s*OF\s*\d{4})", t, re.IGNORECASE).group(1)
        dated = re.search(r"Dated\s*:\s*(.+)", t).group(1).strip()
        conn.execute(
            "INSERT INTO judgments (source_path, forum, case_number, date, full_text, extraction_method) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (f"fixture:{case_no}", forum, case_no, dated, t, "fixture"))
    conn.commit()
    return conn


def fresh_db(texts=None):
    return seed(get_conn(":memory:"), texts)


PENDING = (
    "The complainants booked a residential flat with the opposite party in a project on 29.12.2012 "
    "with a booking amount of Rs.7,50,000/-. An agreement was executed between the parties on 23.4.2013. "
    "The possession was to be delivered within a period of 48 months. The construction is not complete "
    "till date though they have already paid more than Rs.70 lakhs to the developer. The complainants are "
    "seeking refund of the amount paid along with interest and compensation. The OP has resisted the "
    "complaint contending that it is barred by limitation."
)


# --------------------------------------------------------------------------
class TestHelpers(unittest.TestCase):
    def test_forum_hierarchy_is_ordered(self):
        self.assertGreater(authority_weight("Supreme Court"), authority_weight("NCDRC"))
        self.assertGreater(authority_weight("NCDRC"), authority_weight("State Commission"))
        self.assertGreater(authority_weight("State Commission"), authority_weight("District Forum"))
        self.assertEqual(authority_weight(None), 0.5)

    def test_detect_forum_extended_prefers_apex_courts(self):
        self.assertEqual(detect_forum_extended("IN THE SUPREME COURT OF INDIA\nCivil Appeal"), "Supreme Court")
        self.assertEqual(detect_forum_extended(FIXTURES[0]), "NCDRC")
        self.assertEqual(detect_forum_extended(FIXTURES[1]), "State Commission")

    def test_case_signals(self):
        s = case_signals(PENDING)
        self.assertIn("refund_sought", s)
        self.assertIn("limitation_defence", s)
        self.assertNotIn("medical_treatment", s)

    def test_jaccard(self):
        self.assertEqual(jaccard({"a", "b"}, {"a", "b"}), 1.0)
        self.assertEqual(jaccard({"a"}, {"b"}), 0.0)
        self.assertEqual(jaccard(set(), set()), 0.0)

    def test_recency_decays_and_handles_unknown(self):
        recent = recency_score("12 Mar 2019", reference=date(2020, 1, 1))
        old = recency_score("05 Jun 2001", reference=date(2020, 1, 1))
        self.assertGreater(recent, old)
        self.assertEqual(recency_score(None), 0.5)
        self.assertEqual(recency_score("not a date"), 0.5)

    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(SCORING_WEIGHTS.values()), 1.0, places=9)


class TestOutcomeAndHoldings(unittest.TestCase):
    def test_outcome_allowed_with_relief(self):
        o = extract_outcome(FIXTURES[0])
        self.assertIn(o.label, ("allowed", "disposed_with_directions"))
        self.assertEqual(o.favours, "complainant")
        self.assertIn("refund", o.relief_types)
        self.assertIn(6000000, o.amounts_inr)
        self.assertIn(9.0, o.interest_rates)

    def test_outcome_dismissed_grants_nothing(self):
        o = extract_outcome(FIXTURES[1])
        self.assertEqual(o.label, "dismissed")
        self.assertEqual(o.favours, "opposite_party")
        self.assertEqual(o.relief_types, [])
        self.assertEqual(o.amounts_inr, [])

    def test_partly_allowed_beats_allowed(self):
        o = extract_outcome(FIXTURES[2])
        self.assertEqual(o.label, "partly_allowed")
        self.assertEqual(o.favours, "complainant")

    def test_outcome_unknown_on_text_without_an_order(self):
        self.assertEqual(extract_outcome(PENDING).label, "unknown")

    def test_holdings_are_grounded_and_ranked(self):
        hs = extract_holdings(FIXTURES[0])
        self.assertTrue(hs)
        for h in hs:
            self.assertEqual(FIXTURES[0][h.span[0]:h.span[1]], h.text)
            self.assertTrue(h.grounded)
        self.assertTrue(any("deficiency in service" in h.text for h in hs))
        self.assertEqual(hs[0].cue_score, max(h.cue_score for h in hs))


class TestQueryConstruction(unittest.TestCase):
    def test_query_built_from_fact_sheet(self):
        sheet = FactExtractionAgent().extract(PENDING)
        q = build_precedent_query(sheet, "refund of flat")
        self.assertIn("refund", q)
        self.assertTrue(any(w in q for w in ("deficiency", "limitation", "delayed")))
        # tokens are de-duplicated
        self.assertEqual(len(q.split()), len(set(q.split())))

    def test_query_absorbs_statutes_from_agent_2(self):
        q = build_precedent_query(None, "delayed possession",
                                  statutes=[{"section": "2(11)", "act": "Consumer Protection Act, 2019"}])
        self.assertIn("section", q)
        self.assertIn("2(11)", q)


class TestRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn = fresh_db()
        cls.sheet = FactExtractionAgent().extract(PENDING)
        cls.bundle = PrecedentRetrievalAgent(conn=cls.conn, top_k=4).retrieve(
            fact_sheet=cls.sheet, case_text=PENDING, user_query="refund for delayed possession")

    def test_returns_ranked_matches(self):
        self.assertIsInstance(self.bundle, PrecedentBundle)
        self.assertTrue(self.bundle.matches)
        scores = [m.final_score for m in self.bundle.matches]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_on_point_precedent_outranks_off_point_one(self):
        ranked = [m.case_number for m in self.bundle.matches]
        self.assertIn("CONSUMER CASE NO. 111 OF 2018", ranked)
        self.assertLess(ranked.index("CONSUMER CASE NO. 111 OF 2018"),
                        ranked.index("CONSUMER CASE NO. 444 OF 2021"))

    def test_every_score_component_is_reported(self):
        for m in self.bundle.matches:
            self.assertEqual(set(m.score_components) >= set(SCORING_WEIGHTS), True)
            self.assertTrue(m.why_relevant)

    def test_matches_carry_outcome_and_holdings(self):
        top = self.bundle.matches[0]
        self.assertNotEqual(top.outcome.label, "unknown")
        self.assertTrue(top.holdings)

    def test_shared_and_distinguishing_factors(self):
        top = self.bundle.matches[0]
        self.assertTrue(top.shared_signals)
        fridge = next(m for m in self.bundle.matches if m.case_number.startswith("CONSUMER CASE NO. 444"))
        self.assertTrue(fridge.distinguishing_factors)

    def test_aggregated_evidence_for_prediction(self):
        self.assertTrue(self.bundle.outcome_distribution)
        self.assertIsNotNone(self.bundle.complainant_success_rate)
        self.assertIsNotNone(self.bundle.authority_weighted_success_rate)
        self.assertTrue(0.0 <= self.bundle.complainant_success_rate <= 1.0)

    def test_verification_runs_on_every_match(self):
        for m in self.bundle.matches:
            self.assertIsInstance(m.verified, bool)
            self.assertTrue(m.verification_reason)
        self.assertEqual(self.bundle.verified_count + self.bundle.unverified_count,
                         len(self.bundle.matches))

    def test_small_corpus_warning(self):
        self.assertTrue(any("indicative" in w for w in self.bundle.warnings))

    def test_top_k_respected(self):
        b = PrecedentRetrievalAgent(conn=self.conn, top_k=2).retrieve(
            fact_sheet=self.sheet, case_text=PENDING)
        self.assertLessEqual(len(b.matches), 2)


class TestLeakageControl(unittest.TestCase):
    def test_self_excluded_by_judgment_id(self):
        conn = fresh_db()
        b = PrecedentRetrievalAgent(conn=conn).retrieve(
            fact_sheet=FactExtractionAgent().extract(PENDING), case_text=PENDING, self_judgment_id=1)
        self.assertIn(1, b.excluded_judgment_ids)
        self.assertNotIn(1, [m.judgment_id for m in b.matches])

    def test_self_excluded_by_text_overlap(self):
        """The pending case is itself in the corpus: it must not return itself."""
        conn = fresh_db(FIXTURES + [FIXTURES[0]])  # duplicate ingested copy
        sheet = FactExtractionAgent().extract(FIXTURES[0])
        b = PrecedentRetrievalAgent(conn=conn).retrieve(fact_sheet=sheet, case_text=FIXTURES[0])
        self.assertGreaterEqual(len(b.excluded_judgment_ids), 2)
        for m in b.matches:
            self.assertNotIn("CONSUMER CASE NO. 111 OF 2018", m.case_number or "")

    def test_exclusion_can_be_disabled(self):
        conn = fresh_db()
        b = PrecedentRetrievalAgent(conn=conn, exclude_self=False).retrieve(
            fact_sheet=FactExtractionAgent().extract(FIXTURES[0]), case_text=FIXTURES[0])
        self.assertEqual(b.excluded_judgment_ids, [])

    def test_no_outcome_of_the_pending_case_leaks_into_results(self):
        conn = fresh_db(FIXTURES + [FIXTURES[0]])
        b = PrecedentRetrievalAgent(conn=conn).retrieve(
            fact_sheet=FactExtractionAgent().extract(FIXTURES[0]), case_text=FIXTURES[0])
        blob = " ".join(m.outcome.evidence for m in b.matches)
        self.assertNotIn("Rs.60,00,000", blob)


class TestEdgeCases(unittest.TestCase):
    def test_empty_corpus(self):
        b = PrecedentRetrievalAgent(conn=get_conn(":memory:")).retrieve(
            fact_sheet=FactExtractionAgent().extract(PENDING), case_text=PENDING)
        self.assertEqual(b.matches, [])
        self.assertTrue(any("empty" in w.lower() for w in b.warnings))

    def test_single_document_corpus_falls_back_to_bm25(self):
        conn = fresh_db([FIXTURES[0]])
        b = PrecedentRetrievalAgent(conn=conn).retrieve(
            fact_sheet=FactExtractionAgent().extract(PENDING), case_text=PENDING)
        self.assertEqual(b.retrieval_mode, "bm25_only")
        self.assertTrue(b.matches)

    def test_empty_query_is_reported_not_guessed(self):
        b = PrecedentRetrievalAgent(conn=fresh_db()).retrieve(fact_sheet=None, case_text="", user_query="")
        self.assertEqual(b.matches, [])
        self.assertTrue(b.warnings)

    def test_min_score_filters_weak_matches(self):
        b = PrecedentRetrievalAgent(conn=fresh_db(), min_score=0.99).retrieve(
            fact_sheet=FactExtractionAgent().extract(PENDING), case_text=PENDING)
        self.assertEqual(b.matches, [])
        self.assertTrue(any("min_score" in w for w in b.warnings))

    def test_works_without_a_fact_sheet(self):
        b = PrecedentRetrievalAgent(conn=fresh_db()).retrieve(
            case_text=PENDING, user_query="refund delayed possession builder")
        self.assertTrue(b.matches)


class TestPipelineContract(unittest.TestCase):
    def test_state_contract_after_agent_1(self):
        conn = fresh_db()
        state = {"case_text": PENDING, "query": "refund for delayed possession"}
        state = FactExtractionAgent().run(state)
        state = PrecedentRetrievalAgent(conn=conn).run(state)
        for k in ("fact_sheet", "precedent_bundle", "precedents", "precedent_query"):
            self.assertIn(k, state)
        self.assertTrue(state["precedent_bundle"].to_dict()["matches"])

    def test_runs_on_the_real_judgment(self):
        if not SAMPLE.exists():
            self.skipTest("sample judgment not present")
        text = SAMPLE.read_text(encoding="utf-8")
        conn = fresh_db()
        state = PrecedentRetrievalAgent(conn=conn).run(
            FactExtractionAgent().run({"case_text": text, "query": "refund delayed possession"}))
        b = state["precedent_bundle"]
        self.assertTrue(b.matches)
        self.assertEqual(b.matches[0].case_number, "CONSUMER CASE NO. 111 OF 2018")


class _GoodLLM:
    def complete_json(self, system, user):
        return [{"id": 0, "relevance": 0.1, "reason": "weak match"},
                {"id": 1, "relevance": 1.0, "reason": "directly on point"}]


class _BadLLM:
    def complete_json(self, system, user):
        raise RuntimeError("no network")


class _HallucinatingLLM:
    def complete_json(self, system, user):
        return [{"id": 99, "relevance": 1.0, "reason": "invented case"}]


class TestLLMReranking(unittest.TestCase):
    def test_llm_reorders_the_shortlist(self):
        conn = fresh_db()
        sheet = FactExtractionAgent().extract(PENDING)
        b = PrecedentRetrievalAgent(conn=conn, llm=_GoodLLM()).retrieve(fact_sheet=sheet, case_text=PENDING)
        scored = [m for m in b.matches if "llm_relevance" in m.score_components]
        self.assertTrue(scored)
        self.assertTrue(any(m.ranked_by == "rules+llm" for m in b.matches))

    def test_llm_failure_falls_back_to_rules(self):
        conn = fresh_db()
        sheet = FactExtractionAgent().extract(PENDING)
        b = PrecedentRetrievalAgent(conn=conn, llm=_BadLLM()).retrieve(fact_sheet=sheet, case_text=PENDING)
        self.assertTrue(b.matches)
        self.assertTrue(all(m.ranked_by == "rules" for m in b.matches))
        self.assertTrue(any("LLM re-ranking unavailable" in w for w in b.warnings))

    def test_llm_cannot_introduce_a_case(self):
        conn = fresh_db()
        sheet = FactExtractionAgent().extract(PENDING)
        b = PrecedentRetrievalAgent(conn=conn, llm=_HallucinatingLLM()).retrieve(
            fact_sheet=sheet, case_text=PENDING)
        for m in b.matches:
            self.assertIn(m.case_number, [f"CONSUMER CASE NO. {n} OF {y}"
                                          for n, y in (("111", "2018"), ("222", "2015"),
                                                       ("333", "2017"), ("444", "2021"))])


if __name__ == "__main__":
    unittest.main()
