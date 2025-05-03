import importlib
import logging

from .auto_cot import AutoCoT
from .cdw_cot import CDWCoT
from .clustering import BaseClusterer, KMeansClusterer
from .embedding import BaseEmbedder, SentenceTransformerEmbedder
from .graph_of_thoughts import GraphOfThoughts
from .least_to_most import LeastToMost
from .model import BaseLLM, OllamaLLM, OpenAILLM
from .sc_cot import SelfConsistency
from .schemas import (
    EvaluationResult,
    ExtractedAnswer,
    LTMDecomposition,
    ThoughtExpansion,
)
from .tree_of_thoughts import TreeOfThoughts
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
    # Chain of Thought methods and frameworks
    "AutoCoT",
    "CDWCoT",
    "GraphOfThoughts",
    "LeastToMost",
    "SelfConsistency",
    "TreeOfThoughts",
    # Model abstractions
    "BaseLLM",
    "OllamaLLM",
    "OpenAILLM",
    # Schemas
    "LTMDecomposition",
    "ThoughtExpansion",
    "EvaluationResult",
    "ExtractedAnswer",
    # Embedding and Clustering
    "BaseEmbedder",
    "SentenceTransformerEmbedder",
    "BaseClusterer",
    "KMeansClusterer",
    # Utils (bunch of useful stuff)
    "count_steps",
    "approx_token_length",
    "exact_match",
    "accuracy",
]
