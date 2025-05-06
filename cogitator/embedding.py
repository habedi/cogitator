"""Provides abstractions and implementations for text embedding models."""

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


class BaseEmbedder(ABC):
    """Abstract base class for text embedding models."""

    @abstractmethod
    def encode(self, texts: List[str]) -> List[np.ndarray]:
        """Encodes a list of texts into embedding vectors.

        Args:
            texts: A list of strings to encode.

        Returns:
            A list of NumPy arrays, where each array is the embedding vector for
            the corresponding text.
        """
        ...


class SentenceTransformerEmbedder(BaseEmbedder):
    """An embedder implementation using the sentence-transformers library.

    This class uses a singleton pattern to avoid reloading the model multiple times.
    """

    _instance: Optional["SentenceTransformerEmbedder"] = None
    _model: Optional[SentenceTransformer] = None

    def __new__(cls, model_name: str = "all-MiniLM-L6-v2") -> "SentenceTransformerEmbedder":
        """Creates or returns the singleton instance of the embedder.

        Args:
            model_name: The name of the sentence-transformer model to load.
                This argument is only used during the first instantiation.

        Returns:
            The singleton instance of SentenceTransformerEmbedder.
        """
        if cls._instance is None:
            cls._instance = super(SentenceTransformerEmbedder, cls).__new__(cls)
            cls._model = SentenceTransformer(model_name)
        return cls._instance

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initializes the SentenceTransformerEmbedder instance.

        Note: Due to the singleton pattern implemented in `__new__`, the
        `model_name` argument here is effectively ignored after the first
        instantiation. The model loaded is determined by the `model_name`
        passed during the first call to `__new__` or `__init__`.

        Args:
            model_name: The name of the sentence-transformer model. Defaults to
                "all-MiniLM-L6-v2".
        """
        pass

    def encode(self, texts: List[str]) -> List[np.ndarray]:
        """Encodes a list of texts using the loaded sentence-transformer model.

        Args:
            texts: The list of strings to encode.

        Returns:
            A list of NumPy ndarray embeddings.

        Raises:
            RuntimeError: If the embedding model has not been initialized correctly.
        """
        if self._model is None:
            raise RuntimeError("Embedder model not initialized.")
        embeddings: List[np.ndarray] = self._model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        )
        return embeddings
