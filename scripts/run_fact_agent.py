"""
Run the Fact Extraction Agent on a case file.

Usage:
    python scripts/run_fact_agent.py data/ncdrc_judgments/gunjan_aggarwal_257_2019.txt
    python scripts/run_fact_agent.py case.txt --query "refund delayed flat" --json
    python scripts/run_fact_agent.py case.txt --keep-outcome     # do NOT strip the operative order
    python scripts/run_fact_agent.py case.txt --llm              # needs ANTHROPIC_API_KEY
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline.extract import extract
from data_pipeline.clean import clean_text
from agents.fact_extraction import FactExtractionAgent

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--query", default="")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--keep-outcome", action="store_true")
    ap.add_argument("--llm", action="store_true")
    a = ap.parse_args()

    text = clean_text(extract(a.path)["text"])
    llm = None
    if a.llm:
        from agents.llm import AnthropicBackend
        llm = AnthropicBackend()
    agent = FactExtractionAgent(llm=llm, strip_outcome=not a.keep_outcome)
    sheet = agent.extract(text, a.query)
    print(json.dumps(sheet.to_dict(), indent=2, ensure_ascii=False, default=str) if a.json else sheet.summary())
