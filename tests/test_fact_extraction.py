"""Run:  python -m unittest tests.test_fact_extraction -v"""
import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.fact_extraction import (
    FactExtractionAgent, parse_dates, parse_amounts, parse_durations, add_months,
    split_sentences, rule_classify,
)

SAMPLE = Path("data/ncdrc_judgments/gunjan_aggarwal_257_2019.txt")

FREE_TEXT = (
    "I purchased a Samsung refrigerator on 12 March 2024 by paying Rs.45,000/- online. "
    "The compressor became defective within two months and stopped cooling. "
    "The seller failed to repair or replace it despite repeated requests. "
    "I am seeking refund of the price along with compensation of Rs.10,000."
)


class TestHelpers(unittest.TestCase):
    def test_dates(self):
        self.assertEqual(parse_dates("on 23.4.2013 and 10 Jan 2013"), ["2013-04-23", "2013-01-10"])
        self.assertEqual(parse_dates("invalid 31.02.2020"), [])

    def test_amounts(self):
        self.assertEqual(parse_amounts("Rs.7,50,000/- and Rs.70 lakhs and Rs.1.00 crores"),
                         [750000, 7000000, 10000000])

    def test_durations(self):
        d = parse_durations("48 months, six months, 3 1/2 years")
        self.assertEqual([(x["value"], x["unit"]) for x in d], [(48, "month"), (6, "month"), (3.5, "year")])

    def test_add_months_clamps_day(self):
        self.assertEqual(add_months(date(2013, 1, 31), 1), date(2013, 2, 28))
        self.assertEqual(add_months(date(2013, 4, 23), 48), date(2017, 4, 23))

    def test_no_split_on_abbreviations(self):
        s = split_sentences("1. As held in Ambrish Vs. Ferrous Infrastructure Pvt. Ltd. the value is Rs. 95 lacs. Next sentence here is long enough.")
        self.assertEqual(len(s), 2)

    def test_rule_classifier(self):
        self.assertEqual(rule_classify("The complaint has been resisted by the OP which has taken preliminary objections.")[0], "defence")
        self.assertEqual(rule_classify("An agreement was executed between the parties on 23.4.2013.")[0], "transaction")


class TestAgentOnJudgment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SAMPLE.read_text(encoding="utf-8")
        cls.sheet = FactExtractionAgent().extract(cls.text)

    def test_metadata(self):
        self.assertEqual(self.sheet.forum, "NCDRC")
        self.assertEqual(self.sheet.reference_date, "2020-09-28")
        self.assertIn("GUNJAN AGGARWAL", self.sheet.complainants[0])
        self.assertEqual(len(self.sheet.opposite_parties), 2)
        self.assertEqual(self.sheet.dispute_category, "real_estate")

    def test_outcome_not_leaked(self):
        self.assertTrue(self.sheet.outcome_removed)
        everything = " ".join(f.text for f in self.sheet.facts + self.sheet.claims
                              + self.sheet.defences + self.sheet.procedural_points)
        for leaked in ("70,33,103", "9% per annum", "cost of litigation", "within six months from today"):
            self.assertNotIn(leaked, everything)

    def test_keep_outcome_flag(self):
        s = FactExtractionAgent(strip_outcome=False).extract(self.text)
        self.assertFalse(s.outcome_removed)

    def test_every_fact_is_grounded(self):
        for f in self.sheet.facts + self.sheet.claims + self.sheet.defences + self.sheet.procedural_points:
            self.assertEqual(self.text[f.span[0]:f.span[1]], f.text)
            self.assertTrue(f.grounded)

    def test_legal_reasoning_kept_out_of_facts(self):
        self.assertGreater(self.sheet.legal_reasoning_count, 0)
        self.assertFalse(any("Hon'ble Supreme Court" in f.text for f in self.sheet.facts))

    def test_derived_delay(self):
        d = {x.name: x for x in self.sheet.derived_facts}
        self.assertEqual(d["promised_delivery_deadline"].value, "2017-04-23")
        self.assertEqual(int(d["delay_days_at_reference_date"].value), 1254)
        self.assertTrue(all(x.derived for x in self.sheet.derived_facts))

    def test_timeline_sorted(self):
        dates = [e.date for e in self.sheet.timeline]
        self.assertEqual(dates, sorted(dates))
        self.assertIn("2012-12-29", dates)

    def test_hooks_and_query(self):
        self.assertIn("deficiency_in_service", [h["concept"] for h in self.sheet.legal_hooks])
        self.assertIn("refund", self.sheet.retrieval_query)

    def test_pipeline_state_contract(self):
        state = FactExtractionAgent().run({"case_text": self.text, "query": "builder delay"})
        for k in ("fact_sheet", "retrieval_query", "legal_hooks"):
            self.assertIn(k, state)


class TestFreeText(unittest.TestCase):
    def test_free_text_case_file(self):
        s = FactExtractionAgent().extract(FREE_TEXT)
        self.assertGreaterEqual(len(s.facts), 3)
        self.assertEqual(s.claims[0].amounts_inr, [10000])
        self.assertIn("2024-03-12", [e.date for e in s.timeline])
        self.assertIn("defect_in_goods", [h["concept"] for h in s.legal_hooks])
        self.assertTrue(any("Party names not found" in w for w in s.warnings))

    def test_empty_input(self):
        s = FactExtractionAgent().extract("  ")
        self.assertTrue(s.warnings)


class _GoodLLM:
    def complete_json(self, system, user):
        return [{"id": 0, "label": "claim", "confidence": 0.9}]


class _BadLLM:
    def complete_json(self, system, user):
        raise RuntimeError("no network")


class TestLLMBackend(unittest.TestCase):
    def test_llm_relabels(self):
        s = FactExtractionAgent(llm=_GoodLLM()).extract(FREE_TEXT)
        self.assertEqual(s.claims[0].labelled_by, "llm")
        self.assertTrue(s.claims[0].grounded)

    def test_llm_failure_falls_back_to_rules(self):
        s = FactExtractionAgent(llm=_BadLLM()).extract(FREE_TEXT)
        self.assertGreaterEqual(len(s.facts), 3)
        self.assertTrue(all(f.labelled_by == "rules" for f in s.facts))


if __name__ == "__main__":
    unittest.main()
