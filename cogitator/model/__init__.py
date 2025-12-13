"""Provides interfaces and implementations for LLM providers.

This subpackage defines the abstract base class `BaseLLM` and implementations for interacting with
different LLM services like OpenAI, Ollama, and OpenRouter.
"""

from .base import BaseLLM
from .ollama import OllamaLLM
from .openai import OpenAILLM
from .openrouter import OpenRouterLLM

__all__ = [
    "BaseLLM",
    "OllamaLLM",
    "OpenAILLM",
    "OpenRouterLLM",
]
