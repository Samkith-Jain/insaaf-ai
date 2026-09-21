"""
Stage 8: Output & API layer.

Production target (per architecture diagram / requirements.txt): FastAPI.
This sandbox has no network access to pip-install fastapi/uvicorn/pydantic,
so this is built on Python's stdlib http.server instead — same route
contract (GET /api/judgments, GET /api/judgments/{id}, POST /api/query),
same JSON in/out shape. Porting to FastAPI later means moving these handler
bodies into `@app.get(...)` / `@app.post(...)` functions almost unchanged.

Run:
    python -m api.server
Serves on http://localhost:8000
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from structuring.db import get_conn, fetch_all_judgments
from structuring.segment import StructuredJudgment
from retrieval.hybrid import HybridRetriever
from prompt_correction.correct import correct_prompt
from verification.citeverify import verify_all
from explanation.irac import generate_irac
from agents.fact_extraction import FactExtractionAgent
from agents.precedent_retrieval import PrecedentRetrievalAgent
from agents.prediction import PredictionAgent
from agents.statute_retrieval import StatuteRetrievalAgent
from agents.argument_analysis import ArgumentAnalysisAgent

DB_PATH = "data_pipeline/legal.db"
_retriever = None  # built lazily / rebuilt on demand


def get_retriever(conn):
    global _retriever
    documents = fetch_all_judgments(conn)
    if _retriever is None or _retriever.documents != documents:
        _retriever = HybridRetriever(documents) if len(documents) >= 2 else None
    return _retriever, documents


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")  # local-dev CORS
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send_json({}, 200)

    def do_GET(self):
        parsed = urlparse(self.path)
        conn = get_conn(DB_PATH)

        if parsed.path == "/api/judgments":
            docs = fetch_all_judgments(conn)
            summary = [
                {
                    "judgment_id": d["judgment_id"],
                    "case_number": d["case_number"],
                    "forum": d["forum"],
                    "date": d["date"],
                    "snippet": d["full_text"][:200].replace("\n", " "),
                }
                for d in docs
            ]
            self._send_json({"count": len(summary), "judgments": summary})
            return

        if parsed.path.startswith("/api/judgments/") and parsed.path.endswith("/irac"):
            judgment_id = parsed.path.split("/")[-2]
            row = conn.execute(
                "SELECT case_number, full_text FROM judgments WHERE judgment_id = ?", (judgment_id,)
            ).fetchone()
            if not row:
                self._send_json({"error": "not found"}, 404)
                return
            statutes = conn.execute(
                "SELECT section, act FROM statute_citations WHERE judgment_id = ?", (judgment_id,)
            ).fetchall()
            precedents = conn.execute(
                "SELECT case_name, year, reporter_citation FROM precedent_citations WHERE judgment_id = ?",
                (judgment_id,),
            ).fetchall()

            sj = StructuredJudgment(
                case_number=row[0],
                full_text=row[1],
                body_text=row[1],
                statute_citations=[{"section": s[0], "act": s[1]} for s in statutes],
                precedent_citations=[{"case_name": p[0], "year": p[1], "reporter_citation": p[2]} for p in precedents],
            )
            verdicts = verify_all(sj, conn)
            irac = generate_irac(sj, int(judgment_id), verdicts)
            self._send_json({
                "judgment_id": irac.judgment_id,
                "case_number": irac.case_number,
                "issue": irac.issue,
                "rule": irac.rule,
                "application": irac.application,
                "conclusion": irac.conclusion,
                "verified_citation_count": irac.verified_citation_count,
                "unverified_citation_count": irac.unverified_citation_count,
                "citation_verdicts": [
                    {
                        "type": v.citation_type, "citation": v.citation_text,
                        "verified": v.verified, "confidence": v.confidence,
                        "matched_against": v.matched_against, "reason": v.reason,
                    }
                    for v in irac.citation_verdicts
                ],
            })
            return

        if parsed.path.startswith("/api/judgments/"):
            judgment_id = parsed.path.rsplit("/", 1)[-1]
            row = conn.execute(
                "SELECT judgment_id, case_number, forum, date, bench, full_text FROM judgments WHERE judgment_id = ?",
                (judgment_id,),
            ).fetchone()
            if not row:
                self._send_json({"error": "not found"}, 404)
                return
            statutes = conn.execute(
                "SELECT section, act FROM statute_citations WHERE judgment_id = ?", (judgment_id,)
            ).fetchall()
            precedents = conn.execute(
                "SELECT case_name, year, reporter_citation FROM precedent_citations WHERE judgment_id = ?",
                (judgment_id,),
            ).fetchall()
            self._send_json({
                "judgment_id": row[0], "case_number": row[1], "forum": row[2],
                "date": row[3], "bench": row[4], "full_text": row[5],
                "statute_citations": [{"section": s[0], "act": s[1]} for s in statutes],
                "precedent_citations": [{"case_name": p[0], "year": p[1], "reporter_citation": p[2]} for p in precedents],
            })
            return

        self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError:
            self._send_json({"error": "invalid JSON body"}, 400)
            return

        if parsed.path == "/api/query":
            raw_query = payload.get("query", "")
            top_k = int(payload.get("top_k", 5))
            if not raw_query.strip():
                self._send_json({"error": "query is required"}, 400)
                return

            correction = correct_prompt(raw_query)
            conn = get_conn(DB_PATH)
            retriever, docs = get_retriever(conn)
            if retriever is None:
                self._send_json({"error": "index not ready — fewer than 2 judgments ingested"}, 503)
                return
            results = retriever.search(correction.corrected_query, top_k=top_k)

            self._send_json({
                "original_query": correction.original_query,
                "corrected_query": correction.corrected_query,
                "spelling_fixes": correction.spelling_fixes,
                "reformulation_notes": correction.reformulation_notes,
                "results": results,
            })
            return

        if parsed.path == "/api/facts":
            # Agent 1: Fact Extraction. Body: {"case_text": "..."} or {"judgment_id": 1},
            # optional "query" and "strip_outcome" (default true = leakage control).
            case_text = payload.get("case_text", "")
            if not case_text and payload.get("judgment_id") is not None:
                row = get_conn(DB_PATH).execute(
                    "SELECT full_text FROM judgments WHERE judgment_id = ?", (payload["judgment_id"],)
                ).fetchone()
                if not row:
                    self._send_json({"error": "judgment not found"}, 404)
                    return
                case_text = row[0]
            if not case_text.strip():
                self._send_json({"error": "case_text or judgment_id is required"}, 400)
                return
            agent = FactExtractionAgent(strip_outcome=bool(payload.get("strip_outcome", True)))
            sheet = agent.extract(case_text, payload.get("query", ""))
            self._send_json(sheet.to_dict())
            return

        if parsed.path == "/api/arguments":
            # Agent 4: Argument Analysis. Body: {"case_text": "..."} or {"judgment_id": 1},
            # optional "query" and "with_precedents" (default true - Agent 3 supplies the
            # precedential support; set false to build the map from statutes alone).
            conn = get_conn(DB_PATH)
            case_text = payload.get("case_text", "")
            judgment_id = payload.get("judgment_id")
            if not case_text and judgment_id is not None:
                row = conn.execute(
                    "SELECT full_text FROM judgments WHERE judgment_id = ?", (judgment_id,)
                ).fetchone()
                if not row:
                    self._send_json({"error": "judgment not found"}, 404)
                    return
                case_text = row[0]
            if not case_text.strip():
                self._send_json({"error": "case_text or judgment_id is required"}, 400)
                return

            state = {
                "case_text": case_text,
                "query": payload.get("query", ""),
                "judgment_id": judgment_id,
            }
            state = FactExtractionAgent().run(state)
            state = StatuteRetrievalAgent(conn=conn).run(state)
            if payload.get("with_precedents", True):
                state = PrecedentRetrievalAgent(conn=conn).run(state)
            state = ArgumentAnalysisAgent(
                count_unverified=bool(payload.get("count_unverified", False))
            ).run(state)
            self._send_json({
                "argument_map": state["argument_map"].to_dict(),
                "argument_features": state["argument_features"],
            })
            return

        if parsed.path == "/api/predict":
            # Agent 5: Prediction. Body: {"case_text": "..."} or {"judgment_id": 1}, optional
            # "query", "with_precedents" (default true - supplies the precedential prior),
            # "abstain_threshold" (0.0 forces a prediction), "use_precedent_prior" and
            # "count_unverified_prior" (ablations). Chains Agents 1 -> 2 -> 3 -> 4 -> 5.
            conn = get_conn(DB_PATH)
            case_text = payload.get("case_text", "")
            judgment_id = payload.get("judgment_id")
            if not case_text and judgment_id is not None:
                row = conn.execute(
                    "SELECT full_text FROM judgments WHERE judgment_id = ?", (judgment_id,)
                ).fetchone()
                if not row:
                    self._send_json({"error": "judgment not found"}, 404)
                    return
                case_text = row[0]
            if not case_text.strip():
                self._send_json({"error": "case_text or judgment_id is required"}, 400)
                return

            state = {
                "case_text": case_text,
                "query": payload.get("query", ""),
                "judgment_id": judgment_id,
            }
            state = FactExtractionAgent().run(state)
            state = StatuteRetrievalAgent(conn=conn).run(state)
            if payload.get("with_precedents", True):
                state = PrecedentRetrievalAgent(conn=conn).run(state)
            state = ArgumentAnalysisAgent().run(state)

            kwargs = {
                "use_precedent_prior": bool(payload.get("use_precedent_prior", True)),
                "count_unverified_prior": bool(payload.get("count_unverified_prior", False)),
            }
            if payload.get("abstain_threshold") is not None:
                kwargs["abstain_threshold"] = float(payload["abstain_threshold"])
            state = PredictionAgent(**kwargs).run(state)

            self._send_json({
                "prediction": state["prediction"].to_dict(),
                "for_explanation": state["prediction_summary"],
            })
            return

        if parsed.path == "/api/statutes":
            # Agent 2: Statute Retrieval. Body: {"case_text": "..."} or {"judgment_id": 1},
            # optional "query", "top_k", "min_score", "both_acts", and "with_precedents"
            # (chains Agent 3 on the same state, so the statutes feed precedent retrieval).
            conn = get_conn(DB_PATH)
            case_text = payload.get("case_text", "")
            judgment_id = payload.get("judgment_id")
            if not case_text and judgment_id is not None:
                row = conn.execute(
                    "SELECT full_text FROM judgments WHERE judgment_id = ?", (judgment_id,)
                ).fetchone()
                if not row:
                    self._send_json({"error": "judgment not found"}, 404)
                    return
                case_text = row[0]
            if not case_text.strip():
                self._send_json({"error": "case_text or judgment_id is required"}, 400)
                return

            state = {
                "case_text": case_text,
                "query": payload.get("query", ""),
                "judgment_id": judgment_id,
            }
            state = FactExtractionAgent().run(state)
            state = StatuteRetrievalAgent(
                conn=conn,
                top_k=int(payload.get("top_k", 6)),
                min_score=float(payload.get("min_score", 0.15)),
                include_other_act=bool(payload.get("both_acts", False)),
            ).run(state)

            response = {"statutes": state["statute_bundle"].to_dict()}
            if payload.get("with_precedents"):
                state = PrecedentRetrievalAgent(conn=conn).run(state)
                response["precedents"] = state["precedent_bundle"].to_dict()
            self._send_json(response)
            return

        if parsed.path == "/api/precedents":
            # Agent 3: Precedent Retrieval. Body: {"case_text": "..."} or {"judgment_id": 1},
            # optional "query", "top_k", "min_score", and "exclude_self" (default true =
            # leakage control: a stored judgment is never returned as its own precedent).
            conn = get_conn(DB_PATH)
            case_text = payload.get("case_text", "")
            judgment_id = payload.get("judgment_id")
            if not case_text and judgment_id is not None:
                row = conn.execute(
                    "SELECT full_text FROM judgments WHERE judgment_id = ?", (judgment_id,)
                ).fetchone()
                if not row:
                    self._send_json({"error": "judgment not found"}, 404)
                    return
                case_text = row[0]
            if not case_text.strip():
                self._send_json({"error": "case_text or judgment_id is required"}, 400)
                return

            state = {
                "case_text": case_text,
                "query": payload.get("query", ""),
                "judgment_id": judgment_id,
            }
            state = FactExtractionAgent().run(state)
            state = PrecedentRetrievalAgent(
                conn=conn,
                top_k=int(payload.get("top_k", 5)),
                min_score=float(payload.get("min_score", 0.0)),
                exclude_self=bool(payload.get("exclude_self", True)),
            ).run(state)
            self._send_json(state["precedent_bundle"].to_dict())
            return

        self._send_json({"error": "not found"}, 404)

    def log_message(self, fmt, *args):
        print(f"[api] {self.address_string()} - {fmt % args}")


def run(port: int = 8000):
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"Insaaf AI API serving on http://localhost:{port}")
    print("Routes: GET /api/judgments  GET /api/judgments/{id}  GET /api/judgments/{id}/irac")
    print("        POST /api/query  POST /api/facts  POST /api/statutes")
    print("        POST /api/precedents  POST /api/arguments  POST /api/predict")
    server.serve_forever()


if __name__ == "__main__":
    run()
