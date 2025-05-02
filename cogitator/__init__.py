import importlib
import logging

from .auto_cot import AutoCoT
from .cdw_cot import CDWCoT
from .graph_of_thoughts import GraphOfThoughts
from .least_to_most import LeastToMost
from .model import BaseLLM, OllamaLLM, OpenAILLM
from .sc_cot import SelfConsistency
from .tree_of_thoughts import TreeOfThoughts

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
]
