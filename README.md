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
Current phase: implementation in progress.

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
