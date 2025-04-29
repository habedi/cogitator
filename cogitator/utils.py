import re
import threading
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


def count_steps(cot: str) -> int:
    return sum(1 for line in cot.splitlines() if re.match(r"^(\d+[\.\)]|[-*•])\s+", line.strip()))


def approx_token_length(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text))


def exact_match(pred: str, gold: str) -> bool:
    return pred.strip().lower() == gold.strip().lower()


def accuracy(preds: List[str], golds: List[str]) -> float:
    if not golds:
        return 0.0
    matches = sum(exact_match(p, g) for p, g in zip(preds, golds))
    return matches / len(golds)


def cluster_embeddings(
    embs: np.ndarray, n_clusters: int, random_state: int = 33
) -> tuple[np.ndarray, np.ndarray]:
    # Explicitly set n_init for consistent behavior across scikit-learn versions
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(embs)
    return labels, kmeans.cluster_centers_


_embedder: Optional[SentenceTransformer] = None
_embedder_lock = threading.Lock()


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            # Double-check locking pattern
            if _embedder is None:
                _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def encode(texts: list[str]) -> List[np.ndarray]:
    embedder = get_embedder()
    # SentenceTransformer returns List[ndarray] when convert_to_numpy=True
    embeddings: List[np.ndarray] = embedder.encode(
        texts, convert_to_numpy=True, show_progress_bar=False
    )
    return embeddings
