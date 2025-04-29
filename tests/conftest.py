import asyncio
import json
import logging
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional

import numpy as np
import pytest

import cogitator.utils as clust_module
import cogitator.utils as emb_module
from cogitator.model import BaseLLM

logger = logging.getLogger(__name__)


DEFAULT_SYNC_RESPONSE = "SYNC_RESPONSE"
DEFAULT_ASYNC_RESPONSE = "ASYNC_RESPONSE"
DEFAULT_JSON_RESPONSE = {"default": "json_sync"}
DEFAULT_ASYNC_JSON_RESPONSE = {"default": "json_async"}
DEFAULT_FINAL_ANSWER = "FINAL_ANSWER_DEFAULT"
DEFAULT_SUBYoutube = "SUBYoutube_DEFAULT"
DEFAULT_JSON_STEPS = ["step1_default_sync"]
DEFAULT_ASYNC_JSON_STEPS = ["step1_default_async"]
DEFAULT_JSON_SUBQUESTIONS = ["subq1_default_sync"]
DEFAULT_ASYNC_JSON_SUBQUESTIONS = ["subq1_default_async"]
DEFAULT_JSON_EVAL = {"score": 7, "justification": "Default Eval Sync"}
DEFAULT_ASYNC_JSON_EVAL = {"score": 8, "justification": "Default Eval Async"}


class ConfigurableFakeLLM(BaseLLM):


    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config if config is not None else {}
        self.sync_calls: List[Dict[str, Any]] = []
        self.async_calls: List[Dict[str, Any]] = []
        self.responses_map: Dict[str, Any] = self._config.get("responses_map",
                                                              {})

    def _get_response_for_prompt(self, prompt: str, method_type: str) -> Any:

        if prompt in self.responses_map:
            return self.responses_map[prompt]


        if "JSON Steps:" in prompt:
            return self._config.get("json_steps",
                                    DEFAULT_ASYNC_JSON_STEPS if 'async' in method_type else DEFAULT_JSON_STEPS)
        if "JSON Evaluation:" in prompt:
            return self._config.get("json_eval",
                                    DEFAULT_ASYNC_JSON_EVAL if 'async' in method_type else DEFAULT_JSON_EVAL)
        if "JSON Subquestions:" in prompt:
            return self._config.get("json_subquestions",
                                    DEFAULT_ASYNC_JSON_SUBQUESTIONS if 'async' in method_type else DEFAULT_JSON_SUBQUESTIONS)
        if "Current Subquestion:" in prompt:
            return self._config.get("sub_answer", DEFAULT_SUBYoutube + (
                "_async" if 'async' in method_type else ""))
        if "Final Answer:" in prompt or "Given reasoning steps" in prompt:
            return self._config.get("final_answer", DEFAULT_FINAL_ANSWER + (
                "_async" if 'async' in method_type else ""))


        if method_type == "generate":
            return self._config.get("generate_sync", DEFAULT_SYNC_RESPONSE)
        if method_type == "generate_async":
            return self._config.get("generate_async", DEFAULT_ASYNC_RESPONSE)
        if method_type == "generate_json":
            return self._config.get("generate_json", DEFAULT_JSON_RESPONSE)
        if method_type == "generate_json_async":
            return self._config.get("generate_json_async", DEFAULT_ASYNC_JSON_RESPONSE)

        logger.warning(f"No specific response configured for prompt in {method_type}: {prompt}")
        return "UNHANDLED_FAKE_RESPONSE"

    def generate(self, prompt: str, **kwargs: Any) -> str:
        self.sync_calls.append({"type": "generate", "prompt": prompt, "kwargs": kwargs})
        response = self._get_response_for_prompt(prompt, "generate")
        return str(response) if not isinstance(response, (list, dict)) else json.dumps(response)

    async def generate_async(self, prompt: str, **kwargs: Any) -> str:
        self.async_calls.append({"type": "generate_async", "prompt": prompt, "kwargs": kwargs})
        await asyncio.sleep(0.001)
        response = self._get_response_for_prompt(prompt, "generate_async")
        return str(response) if not isinstance(response, (list, dict)) else json.dumps(response)

    def generate_stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        self.sync_calls.append({"type": "stream", "prompt": prompt, "kwargs": kwargs})
        response = self._get_response_for_prompt(prompt, "generate")
        yield str(response) + "_stream"

    async def generate_stream_async(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        self.async_calls.append({"type": "async_stream", "prompt": prompt, "kwargs": kwargs})
        await asyncio.sleep(0.001)
        response = self._get_response_for_prompt(prompt, "generate_async")
        yield str(response) + "_async_stream"

    def generate_json(self, prompt: str, retries: int = 2, **kwargs: Any) -> Any:
        self.sync_calls.append({"type": "json", "prompt": prompt, "kwargs": kwargs})
        response = self._get_response_for_prompt(prompt, "generate_json")
        if isinstance(response, str):
            try:
                return json.loads(response)
            except json.JSONDecodeError as e:
                logger.error(f"FakeLLM sync JSON decode error for: {response}")
                raise e # Re-raise the exception
        return response

    async def generate_json_async(self, prompt: str, retries: int = 2, **kwargs: Any) -> Any:
        self.async_calls.append({"type": "async_json", "prompt": prompt, "kwargs": kwargs})
        await asyncio.sleep(0.001)
        response = self._get_response_for_prompt(prompt, "generate_json_async")
        if isinstance(response, str):
            try:
                return json.loads(response)
            except json.JSONDecodeError as e:
                logger.error(f"FakeLLM async JSON decode error for: {response}")
                raise e # Re-raise the exception
        return response


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
        return [np.array([float(i)]) for i in range(len(texts))]

    monkeypatch.setattr(emb_module, "encode", fake_encode)

    def fake_cluster(embs: np.ndarray, n_clusters: int, random_state: int = 33) -> tuple[np.ndarray, np.ndarray]:

        logger.debug(f"Fake clustering embeddings (shape {embs.shape}) into {n_clusters} clusters")

        embedding_values = embs.flatten()
        labels = (embedding_values % n_clusters).astype(int)

        centers = np.arange(n_clusters).reshape(n_clusters, 1).astype(float) + 0.5
        logger.debug(f"Generated labels: {labels}")
        logger.debug(f"Generated centers: {centers}")
        return labels, centers

    monkeypatch.setattr(clust_module, "cluster_embeddings", fake_cluster)

