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
from retrieval.hybrid import HybridRetriever
from prompt_correction.correct import correct_prompt

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

        self._send_json({"error": "not found"}, 404)

    def log_message(self, fmt, *args):
        print(f"[api] {self.address_string()} - {fmt % args}")


def run(port: int = 8000):
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"Insaaf AI API serving on http://localhost:{port}")
    print("Routes: GET /api/judgments  GET /api/judgments/{id}  POST /api/query")
    server.serve_forever()


if __name__ == "__main__":
    run()
