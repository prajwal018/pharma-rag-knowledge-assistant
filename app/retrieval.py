from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class Retriever:
    """Lightweight local semantic-style baseline using TF-IDF cosine similarity.

    For production, replace this adapter with an approved embedding model/vector DB.
    """
    def __init__(self, documents: list[dict]):
        self.documents = documents
        self.vectorizer = None
        self.matrix = None
        if documents:
            self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            self.matrix = self.vectorizer.fit_transform([d["text"] for d in documents])

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        if not self.documents or not query.strip():
            return []
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        indexes = np.argsort(scores)[::-1][:top_k]
        results = []
        for index in indexes:
            if scores[index] <= 0:
                continue
            item = dict(self.documents[index])
            item["score"] = float(scores[index])
            results.append(item)
        return results
