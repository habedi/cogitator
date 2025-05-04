import importlib
import logging

from .clustering import BaseClusterer, KMeansClusterer
from .embedding import BaseEmbedder, SentenceTransformerEmbedder
from .model import BaseLLM, OllamaLLM, OpenAILLM
from .schemas import (
    EvaluationResult,
    ExtractedAnswer,
    LTMDecomposition,
    ThoughtExpansion,
)
from .strategies import AutoCoT
from .strategies import CDWCoT
from .strategies import GraphOfThoughts
from .strategies import LeastToMost
from .strategies import SelfConsistency
from .strategies import TreeOfThoughts
from .utils import accuracy, approx_token_length, count_steps, exact_match

_logger = logging.getLogger(__name__)
try:
    __version__ = importlib.metadata.version("cogitator")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0-unknown"
    _logger.warning(
        "Could not determine package version using importlib.metadata. "
        "Is the library installed correctly?"
    )

__all__ = [
    "AutoCoT",
    "BaseClusterer",
    "BaseEmbedder",
    "BaseLLM",
    "CDWCoT",
    "EvaluationResult",
    "ExtractedAnswer",
    "GraphOfThoughts",
    "KMeansClusterer",
    "LTMDecomposition",
    "LeastToMost",
    "OllamaLLM",
    "OpenAILLM",
    "SelfConsistency",
    "SentenceTransformerEmbedder",
    "ThoughtExpansion",
    "TreeOfThoughts",
    "accuracy",
    "approx_token_length",
    "count_steps",
    "exact_match",
]
