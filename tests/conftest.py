# tests/conftest.py (Modified)
import asyncio
import json
import logging
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Tuple, \
    Type  # Added Tuple

import numpy as np
import pytest
from pydantic import BaseModel

import cogitator.utils as clust_module
import cogitator.utils as emb_module
from cogitator import BaseLLM
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
DEFAULT_SUBANSWER = "SUBANSWER_DEFAULT"
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

        longest_match_key = None
        for key_fragment in sorted(self.responses_map.keys(), key=len, reverse=True):
            if key_fragment in prompt:
                longest_match_key = key_fragment
                break

        if longest_match_key is not None:
            return self.responses_map[longest_match_key]

        is_json_method = "json" in method_type
        if "JSON Output:" in prompt and "thoughts" in prompt:
            key = "json_steps"
            default = DEFAULT_ASYNC_JSON_STEPS if 'async' in method_type else DEFAULT_JSON_STEPS
        elif "JSON list of strings" in prompt:
            key = "generate_async" if 'async' in method_type else "generate_sync"
            default = json.dumps(
                DEFAULT_ASYNC_JSON_STEPS.model_dump()) if 'async' in method_type else json.dumps(
                DEFAULT_JSON_STEPS.model_dump())
        elif "JSON Output:" in prompt and "subquestions" in prompt:
            key = "json_subquestions"
            default = DEFAULT_ASYNC_JSON_SUBQUESTIONS if 'async' in method_type else DEFAULT_JSON_SUBQUESTIONS
        elif "JSON Evaluation:" in prompt:
            key = "json_eval"
            default = DEFAULT_ASYNC_JSON_EVAL if 'async' in method_type else DEFAULT_JSON_EVAL
        elif "JSON Answer:" in prompt:
            key = "json_answer"

            return self._get_next_response(method_type, key,
                                           DEFAULT_ASYNC_JSON_RESPONSE if 'async' in method_type else DEFAULT_JSON_RESPONSE)
        elif "Current Subquestion:" in prompt:
            key = "sub_answer"
            default = DEFAULT_SUBANSWER + ("_async" if 'async' in method_type else "")
        elif "Given reasoning steps" in prompt \
            or prompt.startswith("Answer the question:") \
            or prompt.startswith(
            "Based on the following sequential subquestions"):
            key = "final_answer"
            default = DEFAULT_FINAL_ANSWER + ("_async" if 'async' in method_type else "")
        else:
            if method_type == "generate":
                key, default = "generate_sync", DEFAULT_SYNC_RESPONSE
            elif method_type == "generate_async":
                key, default = "generate_async", DEFAULT_ASYNC_RESPONSE
            elif method_type == "_generate_json_internal":
                key, default = "generate_json", DEFAULT_JSON_RESPONSE
            elif method_type == "_generate_json_internal_async":
                key, default = "generate_json_async", DEFAULT_ASYNC_JSON_RESPONSE
            elif method_type == "stream":
                key, default = "generate_sync", DEFAULT_SYNC_RESPONSE
            elif method_type == "async_stream":
                key, default = "generate_async", DEFAULT_ASYNC_RESPONSE
            else:
                key, default = "unhandled", "UNHANDLED_FAKE_RESPONSE"

        return self._get_next_response(method_type, key, default)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        self.sync_calls.append({"type": "generate", "prompt": prompt, "kwargs": kwargs})
        response = self._get_response_for_prompt(prompt, "generate")

        if not isinstance(response, str):
            try:

                if isinstance(response, BaseModel): return response.model_dump_json()
                if isinstance(response, (dict, list)): return json.dumps(response)
            except Exception:
                pass
            return str(response)
        return response

    async def generate_async(self, prompt: str, **kwargs: Any) -> str:
        self.async_calls.append({"type": "generate_async", "prompt": prompt, "kwargs": kwargs})
        await asyncio.sleep(0.001)
        response = self._get_response_for_prompt(prompt, "generate_async")

        if not isinstance(response, str):
            try:
                if isinstance(response, BaseModel): return response.model_dump_json()
                if isinstance(response, (dict, list)): return json.dumps(response)
            except Exception:
                pass
            return str(response)
        return response

    def generate_stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        self.sync_calls.append({"type": "stream", "prompt": prompt, "kwargs": kwargs})
        response = self._get_response_for_prompt(prompt, "stream")
        yield str(response) + "_stream"

    async def generate_stream_async(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        self.async_calls.append({"type": "async_stream", "prompt": prompt, "kwargs": kwargs})
        await asyncio.sleep(0.001)
        response = self._get_response_for_prompt(prompt, "async_stream")
        yield str(response) + "_async_stream"

    # --- MODIFICATION START ---
    def _generate_json_internal(self, prompt: str, response_model: Type[BaseModel],
                                **kwargs: Any) -> Tuple[str, Optional[str]]:  # Return tuple
        self.sync_calls.append({
            "type": "_generate_json_internal",
            "prompt": prompt,
            "response_model": response_model.__name__,
            "kwargs": kwargs
        })
        response_obj = self._get_response_for_prompt(prompt, "_generate_json_internal")
        json_string = ""
        if isinstance(response_obj, BaseModel):
            json_string = response_obj.model_dump_json()
        elif isinstance(response_obj, dict):
            json_string = json.dumps(response_obj)
        elif isinstance(response_obj, str):
            try:
                json.loads(response_obj)
                json_string = response_obj
            except json.JSONDecodeError:
                logger.warning(
                    f"Mock Configured string response for JSON prompt is not valid JSON: {response_obj}")
                json_string = "{}"
        else:
            try:
                json_string = json.dumps(response_obj)
            except TypeError:
                logger.warning(
                    f"Mock cannot dump configured response to JSON: {type(response_obj)}")
                json_string = "{}"

        mode_used = "mock_json_mode"  # Return a dummy mode or None
        return json_string, mode_used

    async def _generate_json_internal_async(self, prompt: str, response_model: Type[BaseModel],
                                            **kwargs: Any) -> Tuple[
        str, Optional[str]]:  # Return tuple
        self.async_calls.append({
            "type": "_generate_json_internal_async",
            "prompt": prompt,
            "response_model": response_model.__name__,
            "kwargs": kwargs
        })
        await asyncio.sleep(0.001)
        response_obj = self._get_response_for_prompt(prompt, "_generate_json_internal_async")
        json_string = ""
        if isinstance(response_obj, BaseModel):
            json_string = response_obj.model_dump_json()
        elif isinstance(response_obj, dict):
            json_string = json.dumps(response_obj)
        elif isinstance(response_obj, str):
            try:
                json.loads(response_obj)
                json_string = response_obj
            except json.JSONDecodeError:
                logger.warning(
                    f"Mock Configured string response for async JSON prompt is not valid JSON: {response_obj}")
                json_string = "{}"
        else:
            try:
                json_string = json.dumps(response_obj)
            except TypeError:
                logger.warning(
                    f"Mock cannot dump configured async response to JSON: {type(response_obj)}")
                json_string = "{}"

        mode_used = "mock_json_mode_async"  # Return a dummy mode or None
        return json_string, mode_used
    # --- MODIFICATION END ---


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

        return [np.array([float(i), float(i + 1)], dtype=float) for i in range(len(texts))]

    monkeypatch.setattr(emb_module, "encode", fake_encode)

    def fake_cluster(embs: np.ndarray, n_clusters: int, random_state: int = 33) -> tuple[
        np.ndarray, np.ndarray]:
        logger.debug(f"Fake clustering embeddings (shape {embs.shape}) into {n_clusters} clusters")

        if embs.shape[0] == 0 or n_clusters <= 0:

            output_dim = embs.shape[1] if len(embs.shape) > 1 and embs.shape[1] > 0 else 1
            labels = np.array([], dtype=int)
            centers = np.array([], dtype=float).reshape(0, output_dim)

        else:
            output_dim = embs.shape[1]
            n_clusters = min(n_clusters, embs.shape[0])

            labels = (embs[:, 0] % n_clusters).astype(int)

            centers = np.array(
                [embs[labels == i].mean(axis=0) if np.any(labels == i) else np.zeros(output_dim)
                 for i in range(n_clusters)])

            if centers.ndim == 1 and output_dim > 0:
                centers = centers.reshape(-1, output_dim)
            elif centers.ndim == 0 and output_dim == 0:
                centers = centers.reshape(n_clusters,
                                          1)

        logger.debug(f"Generated labels: {labels}")
        logger.debug(f"Generated centers shape: {centers.shape}")
        return labels, centers

    monkeypatch.setattr(clust_module, "cluster_embeddings", fake_cluster)
