# Insaaf AI

Agentic RAG-powered legal judgment prediction system for Indian consumer dispute adjudication (NCDRC + Consumer Protection Act, 2019).

## Pipeline
1. Prompt Correction — grammar/spell + query reformulation
2. Data Pipeline — OCR (pytesseract), cleaning (pdfplumber)
3. Structuring — segmentation, NER, LLM field extraction
4. Structured DB — SQLite + Qdrant embeddings
5. Hybrid Retrieval — dense (Sentence Transformers) + BM25 + re-ranking
6. Multi-Agent Reasoning — Fact Extraction, Statute Retrieval, Precedent Retrieval, Argument Analysis, Prediction, Verification, Explanation
7. CiteVerify (SPMA) — rule-based citation verification
8. Output — IRAC explanation + FastAPI + React

## Status
Zeroth review: design + literature review complete.
Implemented so far: **stages 1-3** — extraction, cleaning, structuring
(citation NER), SQLite legal DB, hybrid (BM25 + dense) retrieval. Verified
end-to-end against 4 real judgment files (3 NCDRC, 1 State Commission).

Stages 4-8 (multi-agent reasoning, CiteVerify, IRAC explanation, API/frontend)
not yet implemented.

### Note on offline substitutions
This was built in a network-isolated dev environment, so a few components
use dependency-free / offline stand-ins with the same interface as the
production target, swap-in-ready once network access is available:
- **Dense retrieval**: TF-IDF + Truncated SVD (scikit-learn) in place of
  Sentence Transformers + Qdrant — same cosine-similarity ranking interface.
- **NER**: regex-based statute/precedent citation extraction in place of
  spaCy — legal citations follow tight, predictable patterns, so this is a
  reasonable baseline ahead of swapping in a trained NER model.
- **BM25**: implemented from scratch in place of the `rank_bm25` package.

## Setup & run
```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Drop judgment PDFs/HTML/txt into data/raw/ or data/ncdrc_judgments/, then:
python scripts/run_pipeline.py --query "refund for delayed possession of flat by builder"
```
This ingests every file in `data/raw/` and `data/ncdrc_judgments/`, extracts
and structures each judgment into `data_pipeline/legal.db`, builds a hybrid
BM25 + dense index, and prints ranked retrieval results for the query.

### Running the web app (backend + frontend)
```bash
# 1. seed the database (run the pipeline once, or reuse existing legal.db)
python scripts/run_pipeline.py

# 2. start the API server
python -m api.server
# serves on http://localhost:8000 — routes:
#   GET  /api/judgments
#   GET  /api/judgments/{id}
#   POST /api/query   { "query": "...", "top_k": 5 }

# 3. open the frontend — no build step needed
#    just open frontend/index.html directly in a browser,
#    or serve it: python -m http.server 5500 --directory frontend
```
Backend is stdlib `http.server` (not FastAPI — no network access to
pip-install it in this sandbox) and frontend is plain HTML/JS (not React —
no network access to npm-install it). Same route contract / component
boundaries either way, so porting to FastAPI + React is a mechanical swap
once network access is available, not a redesign.

## Structure
- `prompt_correction/` — input correction module
- `data_pipeline/` — OCR, cleaning, raw/processed judgment storage
- `structuring/` — segmentation, NER, field extraction
- `retrieval/` — dense + sparse retrieval, re-ranking
- `agents/` — 7-agent reasoning pipeline
- `verification/` — CiteVerify / SPMA
- `explanation/` — IRAC generation
- `api/` — FastAPI backend
- `frontend/` — React frontend
- `data/` — source judgments, CPA 2019 text
- `tests/`, `scripts/`, `docs/`
