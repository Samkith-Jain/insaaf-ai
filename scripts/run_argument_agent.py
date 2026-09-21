"""
Run the Argument Analysis Agent (Agent 4) on a pending case.

It chains Agents 1 -> 2 -> 3 -> 4, since the argument map is built from all
three upstream bundles.

Usage:
    python scripts/run_argument_agent.py data/ncdrc_judgments/gunjan_aggarwal_257_2019.txt
    python scripts/run_argument_agent.py --judgment-id 1
    python scripts/run_argument_agent.py case.txt --query "refund delayed possession"
    python scripts/run_argument_agent.py case.txt --json
    python scripts/run_argument_agent.py case.txt --no-precedents   # Agents 1 -> 2 -> 4 only
    python scripts/run_argument_agent.py case.txt --count-unverified  # ablation: score unverified support
    python scripts/run_argument_agent.py case.txt --llm             # needs ANTHROPIC_API_KEY

Seed the precedent corpus first with `python scripts/run_pipeline.py`.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.argument_analysis import ArgumentAnalysisAgent
from agents.fact_extraction import FactExtractionAgent
from agents.precedent_retrieval import PrecedentRetrievalAgent
from agents.statute_retrieval import StatuteRetrievalAgent
from data_pipeline.clean import clean_text
from data_pipeline.extract import extract
from structuring.db import get_conn

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="pending case file (.txt/.pdf/.html)")
    ap.add_argument("--judgment-id", type=int, help="pending case already stored in the legal DB")
    ap.add_argument("--db", default="data_pipeline/legal.db")
    ap.add_argument("--query", default="")
    ap.add_argument("--no-precedents", action="store_true", help="skip Agent 3")
    ap.add_argument("--count-unverified", action="store_true",
                    help="ablation: let unverified support contribute to strength scores")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--llm", action="store_true")
    a = ap.parse_args()

    if not a.path and a.judgment_id is None:
        ap.error("give a case file path or --judgment-id")

    conn = get_conn(a.db)
    if a.path:
        case_text = clean_text(extract(a.path)["text"])
    else:
        row = conn.execute("SELECT full_text FROM judgments WHERE judgment_id = ?", (a.judgment_id,)).fetchone()
        if not row:
            sys.exit(f"judgment_id {a.judgment_id} not found in {a.db}")
        case_text = row[0]

    llm = None
    if a.llm:
        from agents.llm import AnthropicBackend
        llm = AnthropicBackend()

    state = {"case_text": case_text, "query": a.query, "judgment_id": a.judgment_id}
    state = FactExtractionAgent().run(state)
    state = StatuteRetrievalAgent(conn=conn).run(state)
    if not a.no_precedents:
        state = PrecedentRetrievalAgent(conn=conn).run(state)
    state = ArgumentAnalysisAgent(llm=llm, count_unverified=a.count_unverified).run(state)

    if a.json:
        print(json.dumps(state["argument_map"].to_dict(), indent=2, ensure_ascii=False, default=str))
    else:
        sheet = state["fact_sheet"]
        print("=== Agents 1-3 (inputs to Agent 4) ===")
        print(f"Facts: {len(sheet.facts)}  claims: {len(sheet.claims)}  defences: {len(sheet.defences)}"
              f"  procedural: {len(sheet.procedural_points)}")
        print(f"Statutes: {[m.section for m in state['statute_bundle'].matches]}")
        if not a.no_precedents:
            print(f"Precedents: {[m.case_number for m in state['precedent_bundle'].matches]}")
        print("\n=== Agent 4: Argument Analysis ===")
        print(state["argument_map"].summary())
