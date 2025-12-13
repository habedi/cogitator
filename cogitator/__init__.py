"""Cogitator: A Python Toolkit for Chain-of-Thought Prompting.

This package provides implementations of various chain-of-thought (CoT) prompting
strategies and frameworks, along with supporting utilities like LLM provider interfaces,
embedding models, clustering algorithms, and data validation schemas.
It aims to make it easier to try and integrate CoT methods into AI applications.
"""

import logging
from importlib.metadata import PackageNotFoundError, version

from .clustering import BaseClusterer, KMeansClusterer
from .embedding import BaseEmbedder, SentenceTransformerEmbedder
from .model import BaseLLM, OllamaLLM, OpenAILLM, OpenRouterLLM
from .schemas import (
    EvaluationResult,
    ExtractedAnswer,
    LTMDecomposition,
    ThoughtExpansion,
)
from .strategies import (
    AutoCoT,
    CDWCoT,
    GraphOfThoughts,
    LeastToMost,
    SelfConsistency,
    TreeOfThoughts,
)
from .utils import accuracy, approx_token_length, count_steps, exact_match

_logger = logging.getLogger(__name__)

try:
    __version__ = version("cogitator")
except PackageNotFoundError:
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
    "OpenRouterLLM",
    "SelfConsistency",
    "SentenceTransformerEmbedder",
    "ThoughtExpansion",
    "TreeOfThoughts",
    "accuracy",
    "approx_token_length",
    "count_steps",
    "exact_match",
]
