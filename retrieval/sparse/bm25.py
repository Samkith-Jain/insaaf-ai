"""
BM25 sparse retrieval, implemented from scratch.
(rank_bm25 is in requirements.txt for production; this sandbox has no network
access to pip-install it, so BM25 is implemented directly here — it's ~40 lines
of well-defined math, not worth blocking the demo over a missing package.)
"""
import math
import re
from collections import Counter


TOKEN_RE = re.compile(r"[a-zA-Z]{2,}")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


class BM25:
    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs_tokens = [tokenize(doc) for doc in corpus]
        self.doc_lens = [len(toks) for toks in self.docs_tokens]
        self.avgdl = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0
        self.doc_freqs = [Counter(toks) for toks in self.docs_tokens]
        self.n_docs = len(corpus)
        self.idf = self._compute_idf()

    def _compute_idf(self) -> dict:
        df = Counter()
        for toks in self.docs_tokens:
            for term in set(toks):
                df[term] += 1
        idf = {}
        for term, freq in df.items():
            idf[term] = math.log(1 + (self.n_docs - freq + 0.5) / (freq + 0.5))
        return idf

    def score(self, query: str, doc_index: int) -> float:
        q_tokens = tokenize(query)
        freqs = self.doc_freqs[doc_index]
        dl = self.doc_lens[doc_index]
        score = 0.0
        for term in q_tokens:
            if term not in freqs:
                continue
            f = freqs[term]
            idf = self.idf.get(term, 0.0)
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
            score += idf * (f * (self.k1 + 1)) / (denom or 1)
        return score

    def rank(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        scores = [(i, self.score(query, i)) for i in range(self.n_docs)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
