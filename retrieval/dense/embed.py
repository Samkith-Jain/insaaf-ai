"""
Dense retrieval.
Production target (per requirements.txt / architecture diagram): Sentence
Transformers embeddings stored in Qdrant. This sandbox has no network access
to download model weights or reach a Qdrant instance, so this module uses
TF-IDF + Truncated SVD (LSA) via scikit-learn as an offline, dependency-free
stand-in that still produces dense vectors + cosine similarity ranking — same
interface, swap-in-ready for SentenceTransformer.encode() + Qdrant search
once network access is available.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


class DenseIndex:
    def __init__(self, corpus: list[str], n_components: int = 100):
        n_components = min(n_components, max(2, len(corpus) - 1))
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        tfidf = self.vectorizer.fit_transform(corpus)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.doc_vectors = self.svd.fit_transform(tfidf)

    def rank(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        q_tfidf = self.vectorizer.transform([query])
        q_vec = self.svd.transform(q_tfidf)
        sims = cosine_similarity(q_vec, self.doc_vectors)[0]
        ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
