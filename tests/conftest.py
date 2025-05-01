import asyncio
import json
import logging
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Type

import numpy as np
import pytest
from pydantic import BaseModel

import cogitator.utils as clust_module
import cogitator.utils as emb_module
from cogitator.model import BaseLLM
from cogitator.schemas import (
    EvaluationResult,
    ExtractedAnswer,
    LTMDecomposition,
    ThoughtExpansion,
)

logger = logging.getLogger(__name__)

DEFAULT_SYNC_RESPONSE = "SYNC_RESPONSE"
DEFAULT_ASYNC_RESPONSE = "ASYNC_RESPONSE"
DEFAULT_JSON_RESPONSE = ExtractedAnswer(final_answer="json_sync_default")
DEFAULT_ASYNC_JSON_RESPONSE = ExtractedAnswer(final_answer="json_async_default")
DEFAULT_FINAL_ANSWER = "FINAL_ANSWER_DEFAULT"
DEFAULT_SUBYoutube = "SUBYoutube_DEFAULT"
DEFAULT_JSON_STEPS = ThoughtExpansion(thoughts=["step1_default_sync"])
DEFAULT_ASYNC_JSON_STEPS = ThoughtExpansion(thoughts=["step1_default_async"])
DEFAULT_JSON_SUBQUESTIONS = LTMDecomposition(subquestions=["subq1_default_sync"])
DEFAULT_ASYNC_JSON_SUBQUESTIONS = LTMDecomposition(subquestions=["subq1_default_async"])
DEFAULT_JSON_EVAL = EvaluationResult(score=7, justification="Default Eval Sync")
DEFAULT_ASYNC_JSON_EVAL = EvaluationResult(score=8, justification="Default Eval Async")


class ConfigurableFakeLLM(BaseLLM):

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config if config is not None else {}
        self.sync_calls: List[Dict[str, Any]] = []
        self.async_calls: List[Dict[str, Any]] = []
        self.responses_map: Dict[str, Any] = self._config.get("responses_map", {})
        self._call_counts = {
            "generate": 0,
            "generate_async": 0,
            "_generate_json_internal": 0,
            "_generate_json_internal_async": 0,
            "stream": 0,
            "async_stream": 0
        }

    def _get_next_response(self, key: str, config_lookup_key: str, default: Any) -> Any:
        if key not in self._call_counts:
            raise KeyError(f"Internal error: key '{key}' not initialized in _call_counts.")

        response_config = self._config.get(config_lookup_key)

        if isinstance(response_config, list):
            if not response_config:
                self._call_counts[key] += 1
                return default
            idx = self._call_counts[key] % len(response_config)
            self._call_counts[key] += 1
            return response_config[idx]
        elif response_config is not None:
            self._call_counts[key] += 1
            return response_config
        else:
            self._call_counts[key] += 1
            return default

    def _get_response_for_prompt(self, prompt: str, method_type: str) -> Any:
        if prompt in self.responses_map:
            return self.responses_map[prompt]

        if "JSON Steps:" in prompt:
            default = DEFAULT_ASYNC_JSON_STEPS if 'async' in method_type else DEFAULT_JSON_STEPS
            return self._get_next_response(method_type, "json_steps", default)
        if "JSON Evaluation:" in prompt:
            default = DEFAULT_ASYNC_JSON_EVAL if 'async' in method_type else DEFAULT_JSON_EVAL
            return self._get_next_response(method_type, "json_eval", default)
        if "JSON Subquestions:" in prompt:
            default = DEFAULT_ASYNC_JSON_SUBQUESTIONS if 'async' in method_type else DEFAULT_JSON_SUBQUESTIONS
            return self._get_next_response(method_type, "json_subquestions", default)
        if "JSON Answer:" in prompt:
            default = DEFAULT_ASYNC_JSON_RESPONSE if 'async' in method_type else DEFAULT_JSON_RESPONSE
            return self._get_next_response(method_type, "json_answer", default)
        if "Current Subquestion:" in prompt:
            default = DEFAULT_SUBYoutube + ("_async" if 'async' in method_type else "")
            return self._get_next_response(method_type, "sub_answer", default)
        if "Final Answer:" in prompt or "Given reasoning steps" in prompt:
            default = DEFAULT_FINAL_ANSWER + ("_async" if 'async' in method_type else "")
            return self._get_next_response(method_type, "final_answer", default)

        if method_type == "generate":
            return self._get_next_response(method_type, "generate_sync", DEFAULT_SYNC_RESPONSE)
        if method_type == "generate_async":
            return self._get_next_response(method_type, "generate_async", DEFAULT_ASYNC_RESPONSE)
        if method_type == "_generate_json_internal":
            return self._get_next_response(method_type, "generate_json", DEFAULT_JSON_RESPONSE)
        if method_type == "_generate_json_internal_async":
            return self._get_next_response(method_type, "generate_json_async",
                                           DEFAULT_ASYNC_JSON_RESPONSE)
        if method_type == "stream":
            return self._get_next_response(method_type, "generate_sync", DEFAULT_SYNC_RESPONSE)
        if method_type == "async_stream":
            return self._get_next_response(method_type, "generate_async", DEFAULT_ASYNC_RESPONSE)

        logger.warning(f"No specific response configured for prompt in {method_type}: {prompt}")
        if method_type in self._call_counts:
            self._call_counts[method_type] += 1
        return "UNHANDLED_FAKE_RESPONSE"

    def generate(self, prompt: str, **kwargs: Any) -> str:
        self.sync_calls.append({"type": "generate", "prompt": prompt, "kwargs": kwargs})
        response = self._get_response_for_prompt(prompt, "generate")

        if "JSON Steps:" in prompt:
            if isinstance(response, BaseModel):
                return response.model_dump_json()
            elif isinstance(response, (dict, list)):
                return json.dumps(response)
            elif isinstance(response, str):
                return response

        return str(response)

    async def generate_async(self, prompt: str, **kwargs: Any) -> str:
        self.async_calls.append({"type": "generate_async", "prompt": prompt, "kwargs": kwargs})
        await asyncio.sleep(0.001)
        response = self._get_response_for_prompt(prompt, "generate_async")

        if "JSON Steps:" in prompt:
            if isinstance(response, BaseModel):
                return response.model_dump_json()
            elif isinstance(response, (dict, list)):
                return json.dumps(response)
            elif isinstance(response, str):
                return response

        return str(response)

    def generate_stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        self.sync_calls.append({"type": "stream", "prompt": prompt, "kwargs": kwargs})
        response = self._get_response_for_prompt(prompt, "stream")
        yield str(response) + "_stream"

    async def generate_stream_async(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        self.async_calls.append({"type": "async_stream", "prompt": prompt, "kwargs": kwargs})
        await asyncio.sleep(0.001)
        response = self._get_response_for_prompt(prompt, "async_stream")
        yield str(response) + "_async_stream"

    def _generate_json_internal(self, prompt: str, response_model: Type[BaseModel],
                                **kwargs: Any) -> str:
        self.sync_calls.append({
            "type": "_generate_json_internal",
            "prompt": prompt,
            "response_model": response_model.__name__,
            "kwargs": kwargs
        })
        response_obj = self._get_response_for_prompt(prompt, "_generate_json_internal")

        if isinstance(response_obj, BaseModel):
            return response_obj.model_dump_json()
        elif isinstance(response_obj, dict):
            return json.dumps(response_obj)
        elif isinstance(response_obj, str):
            return response_obj
        return str(response_obj)

    async def _generate_json_internal_async(self, prompt: str, response_model: Type[BaseModel],
                                            **kwargs: Any) -> str:
        self.async_calls.append({
            "type": "_generate_json_internal_async",
            "prompt": prompt,
            "response_model": response_model.__name__,
            "kwargs": kwargs
        })
        await asyncio.sleep(0.001)
        response_obj = self._get_response_for_prompt(prompt, "_generate_json_internal_async")
        if isinstance(response_obj, BaseModel):
            return response_obj.model_dump_json()
        elif isinstance(response_obj, dict):
            return json.dumps(response_obj)
        elif isinstance(response_obj, str):
            return response_obj
        return str(response_obj)


@pytest.fixture
def fake_llm_factory() -> Callable[[Optional[Dict[str, Any]]], ConfigurableFakeLLM]:
    def _create_llm(config: Optional[Dict[str, Any]] = None) -> ConfigurableFakeLLM:
        return ConfigurableFakeLLM(config)

    return _create_llm


@pytest.fixture
def patch_embedding_clustering(monkeypatch):
    logger.debug("Patching embedding and clustering")

    def fake_encode(texts: List[str]) -> List[np.ndarray]:
        logger.debug(f"Fake encoding texts: {texts}")
        return [np.array([float(i)], dtype=float) for i in range(len(texts))]

    monkeypatch.setattr(emb_module, "encode", fake_encode)

    def fake_cluster(embs: np.ndarray, n_clusters: int, random_state: int = 33) -> tuple[
        np.ndarray, np.ndarray]:
        logger.debug(f"Fake clustering embeddings (shape {embs.shape}) into {n_clusters} clusters")
        embedding_values = embs.flatten()
        if len(embedding_values) == 0 or n_clusters <= 0:
            labels = np.array([], dtype=int)
            centers = np.array([], dtype=float).reshape(0, 1)
        else:
            n_clusters = min(n_clusters, len(embedding_values))
            labels = (embedding_values % n_clusters).astype(int)
            centers = np.array([embedding_values[labels == i].mean() if np.any(labels == i) else 0.0
                                for i in range(n_clusters)])
            centers = centers.reshape(-1, 1)

        logger.debug(f"Generated labels: {labels}")
        logger.debug(f"Generated centers: {centers}")
        return labels, centers

    monkeypatch.setattr(clust_module, "cluster_embeddings", fake_cluster)
