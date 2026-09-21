"""
Run the Statute Retrieval Agent (Agent 2) on a pending case.

It chains Agent 1 -> Agent 2, and with --with-precedents also runs Agent 3,
so you can see the statutes actually feeding precedent retrieval.

Usage:
    python scripts/run_statute_agent.py data/ncdrc_judgments/gunjan_aggarwal_257_2019.txt
    python scripts/run_statute_agent.py --judgment-id 1
    python scripts/run_statute_agent.py case.txt --query "refund delayed possession" --top-k 4
    python scripts/run_statute_agent.py case.txt --both-acts   # also show the other Act's equivalents
    python scripts/run_statute_agent.py case.txt --with-precedents
    python scripts/run_statute_agent.py case.txt --json
    python scripts/run_statute_agent.py case.txt --llm         # needs ANTHROPIC_API_KEY

The legal DB is optional here - it only supplies the corpus-support signal
(how often ingested judgments cite a provision). Seed it with
`python scripts/run_pipeline.py` to switch that signal on.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.fact_extraction import FactExtractionAgent
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
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--min-score", type=float, default=0.15)
    ap.add_argument("--both-acts", action="store_true", help="include provisions of the Act not in force")
    ap.add_argument("--with-precedents", action="store_true", help="also run Agent 3 on the result")
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
    state = StatuteRetrievalAgent(
        conn=conn, top_k=a.top_k, min_score=a.min_score,
        include_other_act=a.both_acts, llm=llm,
    ).run(state)

    if a.with_precedents:
        from agents.precedent_retrieval import PrecedentRetrievalAgent
        state = PrecedentRetrievalAgent(conn=conn).run(state)

    if a.json:
        out = {"statutes": state["statute_bundle"].to_dict()}
        if a.with_precedents:
            out["precedents"] = state["precedent_bundle"].to_dict()
        print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    else:
        sheet = state["fact_sheet"]
        print("=== Agent 1: Fact Extraction (input to Agent 2) ===")
        print(f"Category: {sheet.dispute_category}  |  hooks: {[h['concept'] for h in sheet.legal_hooks]}")
        print(f"Facts: {len(sheet.facts)}  claims: {len(sheet.claims)}  defences: {len(sheet.defences)}\n")
        print("=== Agent 2: Statute Retrieval ===")
        print(state["statute_bundle"].summary())
        if a.with_precedents:
            print("\n=== Agent 3: Precedent Retrieval (using those statutes) ===")
            print(state["precedent_bundle"].summary())
