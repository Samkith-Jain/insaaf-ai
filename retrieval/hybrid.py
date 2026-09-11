"""
Stage 3: Hybrid Retrieval.
Combines sparse (BM25) and dense (TF-IDF+SVD stand-in) rankings via
min-max normalized weighted score fusion, then re-ranks.
"""
from retrieval.sparse.bm25 import BM25
from retrieval.dense.embed import DenseIndex


def _normalize(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return scores
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class HybridRetriever:
    def __init__(self, documents: list[dict], dense_weight: float = 0.5):
        """documents: list of {"judgment_id", "case_number", "full_text", ...}"""
        self.documents = documents
        corpus = [d["full_text"] for d in documents]
        self.bm25 = BM25(corpus)
        self.dense = DenseIndex(corpus)
        self.dense_weight = dense_weight

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        n = len(self.documents)
        bm25_scores = dict(self.bm25.rank(query, top_k=n))
        dense_scores = dict(self.dense.rank(query, top_k=n))

        bm25_norm = _normalize(bm25_scores)
        dense_norm = _normalize(dense_scores)

        fused = {}
        for i in range(n):
            fused[i] = (1 - self.dense_weight) * bm25_norm.get(i, 0.0) + self.dense_weight * dense_norm.get(i, 0.0)

        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for idx, score in ranked:
            doc = self.documents[idx]
            results.append({
                "judgment_id": doc.get("judgment_id"),
                "case_number": doc.get("case_number"),
                "forum": doc.get("forum"),
                "hybrid_score": round(score, 4),
                "bm25_score": round(bm25_scores.get(idx, 0.0), 4),
                "dense_score": round(dense_scores.get(idx, 0.0), 4),
                "snippet": doc["full_text"][:250].replace("\n", " ") + "...",
            })
        return results
