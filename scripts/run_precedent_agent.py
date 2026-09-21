"""
Run the Precedent Retrieval Agent (Agent 3) on a pending case.

It chains Agent 1 -> Agent 3, so the precedents are retrieved from the
structured fact sheet, exactly as they are in the pipeline.

Usage:
    # pending case from a file, precedents from the ingested legal DB
    python scripts/run_precedent_agent.py data/ncdrc_judgments/gunjan_aggarwal_257_2019.txt

    # pending case already ingested — pass its id so it is excluded from its own results
    python scripts/run_precedent_agent.py --judgment-id 1

    python scripts/run_precedent_agent.py case.txt --query "refund delayed possession" --top-k 3
    python scripts/run_precedent_agent.py case.txt --json
    python scripts/run_precedent_agent.py case.txt --keep-self    # disable leakage control (debug only)
    python scripts/run_precedent_agent.py case.txt --llm          # needs ANTHROPIC_API_KEY

Seed the DB first:  python scripts/run_pipeline.py
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.fact_extraction import FactExtractionAgent
from agents.precedent_retrieval import PrecedentRetrievalAgent
from data_pipeline.clean import clean_text
from data_pipeline.extract import extract
from structuring.db import get_conn

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="pending case file (.txt/.pdf/.html)")
    ap.add_argument("--judgment-id", type=int, help="pending case already stored in the legal DB")
    ap.add_argument("--db", default="data_pipeline/legal.db")
    ap.add_argument("--query", default="")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--keep-self", action="store_true", help="do NOT exclude the case from its own results")
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
    state = PrecedentRetrievalAgent(
        conn=conn, top_k=a.top_k, min_score=a.min_score,
        exclude_self=not a.keep_self, llm=llm,
    ).run(state)

    bundle = state["precedent_bundle"]
    if a.json:
        print(json.dumps(bundle.to_dict(), indent=2, ensure_ascii=False, default=str))
    else:
        print("=== Agent 1: Fact Extraction (input to Agent 3) ===")
        sheet = state["fact_sheet"]
        print(f"Category: {sheet.dispute_category}  |  hooks: {[h['concept'] for h in sheet.legal_hooks]}")
        print(f"Facts: {len(sheet.facts)}  claims: {len(sheet.claims)}  defences: {len(sheet.defences)}\n")
        print("=== Agent 3: Precedent Retrieval ===")
        print(bundle.summary())
