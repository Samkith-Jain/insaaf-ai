"""
End-to-end demo: runs the full data pipeline (stages 1-3 of the architecture
diagram) over every file in data/raw/ and data/ncdrc_judgments/, then runs a
sample hybrid retrieval query against the resulting index.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --query "refund for delayed possession builder"
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline.extract import extract
from data_pipeline.clean import clean_text
from structuring.segment import segment
from structuring.db import get_conn, insert_judgment, fetch_all_judgments
from retrieval.hybrid import HybridRetriever
from prompt_correction.correct import correct_prompt
from verification.citeverify import verify_all
from explanation.irac import generate_irac, render_irac

RAW_DIRS = ["data/raw", "data/ncdrc_judgments"]
SUPPORTED_EXT = {".pdf", ".html", ".htm", ".txt"}


def ingest_all(conn):
    files = []
    for d in RAW_DIRS:
        p = Path(d)
        if p.exists():
            files.extend([f for f in p.iterdir() if f.suffix.lower() in SUPPORTED_EXT])

    print(f"Found {len(files)} source file(s) to ingest.\n")
    for f in files:
        print(f"--- {f.name} ---")
        try:
            result = extract(str(f))
        except Exception as e:
            print(f"  [SKIP] extraction failed: {e}\n")
            continue
        cleaned = clean_text(result["text"])
        sj = segment(cleaned)
        judgment_id = insert_judgment(conn, sj, str(f), result["method"])
        print(f"  forum={sj.forum!r}  case_no={sj.case_number!r}  date={sj.date!r}")
        print(f"  method={result['method']}  chars={len(cleaned)}")
        print(f"  statutes_found={len(sj.statute_citations)}  precedents_found={len(sj.precedent_citations)}")
        if sj.statute_citations:
            print(f"    e.g. {sj.statute_citations[0]}")
        if sj.precedent_citations:
            print(f"    e.g. {sj.precedent_citations[0]['case_name']}")
        print(f"  -> stored as judgment_id={judgment_id}\n")


def run_query(conn, raw_query: str, top_k: int = 5):
    correction = correct_prompt(raw_query)
    print(f"\n=== Stage 0: Prompt Correction ===")
    print(correction.summary())

    documents = fetch_all_judgments(conn)
    if len(documents) < 2:
        print("Need at least 2 ingested documents to build a retrieval index.")
        return
    retriever = HybridRetriever(documents)
    results = retriever.search(correction.corrected_query, top_k=top_k)
    print(f"\n=== Stage 3: Hybrid retrieval results ===")
    for r in results:
        print(f"[{r['hybrid_score']}] (bm25={r['bm25_score']} dense={r['dense_score']}) "
              f"{r['case_number']} ({r['forum']})")
        print(f"    {r['snippet']}\n")

    # Stage 7+4b: CiteVerify + IRAC explanation on the top-ranked judgment
    if results:
        top_id = results[0]["judgment_id"]
        top_doc = next(d for d in documents if d["judgment_id"] == top_id)
        top_sj = segment(top_doc["full_text"])
        verdicts = verify_all(top_sj, conn)
        irac = generate_irac(top_sj, top_id, verdicts)
        print(f"\n=== Stage 7: CiteVerify + Stage 4b: IRAC Explanation (top result) ===")
        print(render_irac(irac))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="flat posession delay refund")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    conn = get_conn("data_pipeline/legal.db")
    ingest_all(conn)
    run_query(conn, args.query, args.top_k)
