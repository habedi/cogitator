import abc
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


class BaseEmbedder(abc.ABC):
    @abc.abstractmethod
    def encode(self, texts: List[str]) -> List[np.ndarray]: ...


class SentenceTransformerEmbedder(BaseEmbedder):
    _instance: Optional["SentenceTransformerEmbedder"] = None
    _model: Optional[SentenceTransformer] = None

    def __new__(cls, model_name: str = "all-MiniLM-L6-v2"):
        if cls._instance is None:
            cls._instance = super(SentenceTransformerEmbedder, cls).__new__(cls)
            cls._model = SentenceTransformer(model_name)
        return cls._instance

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        pass

    def encode(self, texts: List[str]) -> List[np.ndarray]:
        if self._model is None:
            raise RuntimeError("Embedder model not initialized.")
        embeddings: List[np.ndarray] = self._model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        )
        return embeddings
