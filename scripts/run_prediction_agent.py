"""
Run the Prediction Agent (Agent 5) on a pending case.

It chains Agents 1 -> 2 -> 3 -> 4 -> 5, since the prediction is built on the
argument map, the statutory findings and the precedential prior.

Usage:
    python scripts/run_prediction_agent.py data/ncdrc_judgments/gunjan_aggarwal_257_2019.txt
    python scripts/run_prediction_agent.py --judgment-id 1
    python scripts/run_prediction_agent.py case.txt --json
    python scripts/run_prediction_agent.py case.txt --no-precedents     # ablate the prior
    python scripts/run_prediction_agent.py case.txt --no-prior          # neutral prior only
    python scripts/run_prediction_agent.py case.txt --abstain 0.0       # force a prediction
    python scripts/run_prediction_agent.py case.txt --llm               # needs ANTHROPIC_API_KEY

    # Backtest: predict a DECIDED judgment with its operative order stripped,
    # then compare against what the tribunal actually held.
    python scripts/run_prediction_agent.py --backtest --judgment-id 1
    python scripts/run_prediction_agent.py --backtest-all

Seed the corpus first with `python scripts/run_pipeline.py`.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.argument_analysis import ArgumentAnalysisAgent
from agents.fact_extraction import FactExtractionAgent
from agents.precedent_retrieval import PrecedentRetrievalAgent, extract_outcome
from agents.prediction import PredictionAgent, score_against_actual
from agents.statute_retrieval import StatuteRetrievalAgent
from data_pipeline.clean import clean_text
from data_pipeline.extract import extract
from structuring.db import get_conn


def build_state(case_text, query, conn, judgment_id=None,
                with_precedents=True, llm=None, agent=None):
    """Agents 1 -> 2 -> 3 -> 4 -> 5 on one shared state dict."""
    state = {"case_text": case_text, "query": query, "judgment_id": judgment_id}
    state = FactExtractionAgent().run(state)
    state = StatuteRetrievalAgent(conn=conn).run(state)
    if with_precedents:
        state = PrecedentRetrievalAgent(conn=conn).run(state)
    state = ArgumentAnalysisAgent(llm=llm).run(state)
    return (agent or PredictionAgent(llm=llm)).run(state)


def actual_outcome(case_text: str) -> str:
    """
    Read the real disposition off a decided judgment, for backtesting only.

    This uses the SAME outcome reader Agent 3 applies to precedents, so the
    ground-truth label and the predicted label come from one vocabulary. It is
    called only after the prediction has been made, and never inside the
    pipeline - the agents themselves never see this.
    """
    return extract_outcome(case_text).label


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="pending case file (.txt/.pdf/.html)")
    ap.add_argument("--judgment-id", type=int, help="case already stored in the legal DB")
    ap.add_argument("--db", default="data_pipeline/legal.db")
    ap.add_argument("--query", default="")
    ap.add_argument("--no-precedents", action="store_true", help="skip Agent 3 entirely")
    ap.add_argument("--no-prior", action="store_true",
                    help="ablation: keep Agent 3 but predict from a neutral prior")
    ap.add_argument("--count-unverified", action="store_true",
                    help="ablation: let unverified precedents into the prior")
    ap.add_argument("--abstain", type=float, default=None,
                    help="abstention threshold (0.0 forces a prediction on every case)")
    ap.add_argument("--backtest", action="store_true",
                    help="compare the prediction against the judgment's real disposition")
    ap.add_argument("--backtest-all", action="store_true",
                    help="backtest every judgment in the DB and report aggregate accuracy")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--llm", action="store_true")
    a = ap.parse_args()

    if not a.path and a.judgment_id is None and not a.backtest_all:
        ap.error("give a case file path, --judgment-id, or --backtest-all")

    conn = get_conn(a.db)
    llm = None
    if a.llm:
        from agents.llm import AnthropicBackend
        llm = AnthropicBackend()

    kwargs = dict(llm=llm, use_precedent_prior=not a.no_prior,
                  count_unverified_prior=a.count_unverified)
    if a.abstain is not None:
        kwargs["abstain_threshold"] = a.abstain
    agent = PredictionAgent(**kwargs)

    # ---- aggregate backtest over the whole corpus --------------------------
    if a.backtest_all:
        rows = conn.execute("SELECT judgment_id, case_number, full_text FROM judgments").fetchall()
        results = []
        for jid, case_no, text in rows:
            state = build_state(text, a.query, conn, judgment_id=jid,
                                with_precedents=not a.no_precedents, llm=llm, agent=agent)
            truth = actual_outcome(text)
            if truth == "unknown":
                print(f"[skip] judgment {jid} ({case_no}): no readable disposition")
                continue
            r = score_against_actual(state["prediction"], truth)
            r["judgment_id"], r["case_number"] = jid, case_no
            results.append(r)
            print(f"[{jid}] {case_no}: predicted {r['predicted']:<18} actual {r['actual']:<18} "
                  f"direction={'OK ' if r['direction_correct'] else 'MISS' if r['direction_correct'] is False else 'ABST'} "
                  f"conf={r['confidence']:.2f} brier={r['brier']:.3f}")

        scored = [r for r in results if r["direction_correct"] is not None]
        print(f"\n--- BACKTEST OVER {len(results)} JUDGMENT(S) ---")
        if scored:
            print(f"  coverage (not abstained): {len(scored)}/{len(results)} "
                  f"({len(scored) / len(results):.0%})")
            print(f"  direction accuracy: {sum(r['direction_correct'] for r in scored)}/{len(scored)}")
            print(f"  exact-disposition accuracy: {sum(r['exact_correct'] for r in scored)}/{len(scored)}")
            print(f"  mean Brier score: {sum(r['brier'] for r in scored) / len(scored):.4f}")
        else:
            print("  every case was abstained on - nothing to score.")
        print("\nNOTE: this corpus is tiny and these judgments are also the precedent pool, so "
              "these figures are a smoke test of the harness, not a result. Report accuracy only "
              "on a held-out split.")
        sys.exit(0)

    # ---- single case --------------------------------------------------------
    if a.path:
        case_text = clean_text(extract(a.path)["text"])
    else:
        row = conn.execute("SELECT full_text FROM judgments WHERE judgment_id = ?",
                           (a.judgment_id,)).fetchone()
        if not row:
            sys.exit(f"judgment_id {a.judgment_id} not found in {a.db}")
        case_text = row[0]

    state = build_state(case_text, a.query, conn, judgment_id=a.judgment_id,
                        with_precedents=not a.no_precedents, llm=llm, agent=agent)
    pred = state["prediction"]

    if a.json:
        print(json.dumps(pred.to_dict(), indent=2, ensure_ascii=False, default=str))
    else:
        sheet = state["fact_sheet"]
        print("=== Agents 1-4 (inputs to Agent 5) ===")
        print(f"Facts: {len(sheet.facts)}  claims: {len(sheet.claims)}  "
              f"defences: {len(sheet.defences)}  procedural: {len(sheet.procedural_points)}")
        print(f"Statutes: {[m.section for m in state['statute_bundle'].matches]}")
        if not a.no_precedents:
            pb = state["precedent_bundle"]
            print(f"Precedents: {[m.case_number for m in pb.matches]}  "
                  f"(verified {pb.verified_count}, prior rate {pb.authority_weighted_success_rate})")
        amap = state["argument_map"]
        print(f"Argument map: {len(amap.issues)} issue(s), merits balance "
              f"{amap.merits_balance:+.3f}, threshold risk {amap.threshold_risk:.3f}")
        print("\n=== Agent 5: Prediction ===")
        print(pred.summary())

    if a.backtest:
        truth = actual_outcome(case_text)
        print("\n=== BACKTEST ===")
        if truth == "unknown":
            print("  No readable disposition in this file - nothing to compare against.")
        else:
            print(json.dumps(score_against_actual(pred, truth), indent=2))
            if pred.leakage_suspect:
                print("  ** the operative order was not stripped - this comparison is not valid **")
