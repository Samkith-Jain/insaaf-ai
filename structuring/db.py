"""
Stage 2b: Structured Legal Database (SQLite).
Stores structured judgment records + extracted citations, keyed by judgment_id,
which the retrieval layer (Qdrant/BM25 index) references.
"""
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS judgments (
    judgment_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    source_path     TEXT NOT NULL,
    forum           TEXT,
    case_number     TEXT,
    date            TEXT,
    bench           TEXT,
    complainant     TEXT,
    opposite_party  TEXT,
    full_text       TEXT NOT NULL,
    extraction_method TEXT
);

CREATE TABLE IF NOT EXISTS statute_citations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    judgment_id     INTEGER NOT NULL REFERENCES judgments(judgment_id),
    section         TEXT,
    act             TEXT
);

CREATE TABLE IF NOT EXISTS precedent_citations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    judgment_id     INTEGER NOT NULL REFERENCES judgments(judgment_id),
    case_name       TEXT,
    year            TEXT,
    reporter_citation TEXT
);

CREATE INDEX IF NOT EXISTS idx_statute_act ON statute_citations(act);
CREATE INDEX IF NOT EXISTS idx_precedent_name ON precedent_citations(case_name);
"""


def get_conn(db_path: str = "data_pipeline/legal.db") -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    return conn


def insert_judgment(conn: sqlite3.Connection, sj, source_path: str, extraction_method: str) -> int:
    cur = conn.execute(
        """INSERT INTO judgments
           (source_path, forum, case_number, date, bench, complainant, opposite_party, full_text, extraction_method)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            source_path,
            sj.forum,
            sj.case_number,
            sj.date,
            ", ".join(sj.bench),
            sj.parties.get("complainant_block", "")[:500],
            sj.parties.get("opposite_party_block", "")[:500],
            sj.full_text,
            extraction_method,
        ),
    )
    judgment_id = cur.lastrowid
    for s in sj.statute_citations:
        conn.execute(
            "INSERT INTO statute_citations (judgment_id, section, act) VALUES (?, ?, ?)",
            (judgment_id, s["section"], s["act"]),
        )
    for p in sj.precedent_citations:
        conn.execute(
            "INSERT INTO precedent_citations (judgment_id, case_name, year, reporter_citation) VALUES (?, ?, ?, ?)",
            (judgment_id, p["case_name"], p.get("year"), p.get("reporter_citation")),
        )
    conn.commit()
    return judgment_id


def fetch_all_judgments(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT judgment_id, case_number, forum, date, full_text FROM judgments").fetchall()
    return [
        {"judgment_id": r[0], "case_number": r[1], "forum": r[2], "date": r[3], "full_text": r[4]}
        for r in rows
    ]
