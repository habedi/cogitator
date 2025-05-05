"""Implements the Automatic chain-of-thought (Auto-CoT) strategy."""

import asyncio
import logging
from typing import List, Optional, Tuple, Any

import numpy as np

from ..clustering import BaseClusterer, KMeansClusterer
from ..embedding import BaseEmbedder, SentenceTransformerEmbedder
from ..model import BaseLLM
from ..utils import approx_token_length, count_steps

logger = logging.getLogger(__name__)


class AutoCoT:
    """Implements the Automatic Chain-of-Thought (Auto-CoT) prompting strategy.

    Auto-CoT aims to automatically construct demonstrations for few-shot CoT prompting
    by clustering questions and selecting diverse examples, then generating CoT
    reasoning for them using zero-shot prompts.

    Reference:
        Zhang, Z., Zhang, A., Li, M., & Smola, A. (2022).
        Automatic Chain of Thought Prompting in Large Language Models.
        arXiv preprint arXiv:2210.03493.
    """

    def __init__(
        self,
        llm: BaseLLM,
        n_demos: int = 8,
        max_q_tokens: int = 60,
        max_steps: int = 5,
        *,
        prompt_template: str = "Let's think step by step.",
        max_retries: int = 2,
        max_tokens: Optional[int] = None,
        rand_seed: Optional[int] = None,
        embedder: Optional[BaseEmbedder] = None,
        clusterer: Optional[BaseClusterer] = None,
    ) -> None:
        """Initializes the AutoCoT strategy handler.

        Args:
            llm: The language model instance to use for generation.
            n_demos: The desired number of demonstrations to generate.
            max_q_tokens: Maximum approximate token length for questions selected as demos.
            max_steps: Maximum number of reasoning steps allowed in a generated demo CoT.
            prompt_template: The zero-shot prompt template used to generate CoT reasoning.
            max_retries: Maximum number of retries for generating a CoT demo if LLM fails.
            max_tokens: Maximum tokens for LLM generation calls (demos and final answer).
            rand_seed: Random seed for clustering and potential LLM seeding.
            embedder: The embedding model instance. Defaults to SentenceTransformerEmbedder.
            clusterer: The clustering algorithm instance. Defaults to KMeansClusterer.
        """
        self.llm = llm
        self.n_demos = n_demos
        self.max_q_tokens = max_q_tokens
        self.max_steps = max_steps
        self.prompt_template = prompt_template
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.rand_seed = rand_seed
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.clusterer = clusterer or KMeansClusterer()
        self.demos: Optional[List[str]] = None

    def fit(self, questions: List[str]) -> None:
        """Builds the demonstration pool using the Auto-CoT process.

        This involves embedding questions, clustering them, selecting diverse
        representatives, generating CoT reasoning for them, and filtering based
        on length and step count criteria.

        Args:
            questions: A list of questions to build demonstrations from.

        Raises:
            ValueError: If the number of questions is less than `n_demos`.
            RuntimeError: If embedding or clustering fails, or if no valid demos
                can be generated.
        """
        if len(questions) < self.n_demos:
            raise ValueError(f"Need >= {self.n_demos} questions, got {len(questions)}")

        embs_list = self.embedder.encode(questions)
        if len(embs_list) == 0:
            raise RuntimeError("Embedding failed to produce results.")
        embs = np.stack(embs_list)

        labels, centers = self.clusterer.cluster(
            embs, self.n_demos, random_seed=self.rand_seed or 0
        )

        candidate_demos: List[Tuple[int, str]] = []
        for c in range(self.n_demos):
            idxs = np.where(labels == c)[0]
            if idxs.size == 0:
                continue
            dists = np.linalg.norm(embs[idxs] - centers[c], axis=1)
            for idx in idxs[np.argsort(dists)]:
                q = questions[idx]
                if approx_token_length(q) > self.max_q_tokens:
                    continue
                candidate_demos.append((idx, q))
                break

        demos: List[str] = []
        for idx, q in candidate_demos:
            prompt = f"Q: {q}\nA: {self.prompt_template}"
            cot: Optional[str] = None
            for attempt in range(self.max_retries + 1):
                try:
                    cot = self.llm.generate(
                        prompt,
                        max_tokens=self.max_tokens,
                        seed=self.rand_seed,
                    )
                    break
                except Exception as e:
                    logger.warning("Retry %d for candidate demo '%s': %s", attempt + 1, q, e)
            if cot is None:
                logger.error(
                    "Failed to generate demo for '%s' after %d retries", q, self.max_retries + 1
                )
                continue
            if count_steps(cot) <= self.max_steps:
                demos.append(f"Q: {q}\nA: {cot}")

        if len(demos) < self.n_demos:
            logger.warning(
                "Could only build %d demos; need %d. Proceeding with available demos.",
                len(demos),
                self.n_demos,
            )
        if not demos:
            raise RuntimeError("Failed to build any valid demos.")

        self.demos = demos

    async def fit_async(
        self, questions: List[str], semaphore: Optional[asyncio.Semaphore] = None
    ) -> None:
        """Asynchronously builds the demonstration pool using the Auto-CoT process.

        Similar to `fit`, but performs LLM generation calls asynchronously.

        Args:
            questions: A list of questions to build demonstrations from.
            semaphore: An optional asyncio.Semaphore to limit concurrent LLM calls.

        Raises:
            ValueError: If the number of questions is less than `n_demos`.
            RuntimeError: If embedding or clustering fails, or if no valid demos
                can be generated.
        """
        if len(questions) < self.n_demos:
            raise ValueError(f"Need >= {self.n_demos} questions, got {len(questions)}")

        embs_list = self.embedder.encode(questions)
        if len(embs_list) == 0:
            raise RuntimeError("Embedding failed to produce results.")
        embs = np.stack(embs_list)

        labels, centers = self.clusterer.cluster(
            embs, self.n_demos, random_seed=self.rand_seed or 0
        )

        candidate_demos_info: List[Tuple[int, str]] = []
        for c in range(self.n_demos):
            idxs = np.where(labels == c)[0]
            if idxs.size == 0:
                continue
            dists = np.linalg.norm(embs[idxs] - centers[c], axis=1)
            for idx in idxs[np.argsort(dists)]:
                q = questions[idx]
                if approx_token_length(q) > self.max_q_tokens:
                    continue
                candidate_demos_info.append((idx, q))
                break

        async def generate_demo(idx: int, q: str) -> Tuple[int, str, Optional[str]]:
            prompt = f"Q: {q}\nA: {self.prompt_template}"
            for attempt in range(self.max_retries + 1):
                try:
                    gen_args = {
                        "max_tokens": self.max_tokens,
                        "seed": self.rand_seed,
                    }
                    if semaphore:
                        async with semaphore:
                            cot = await self.llm.generate_async(prompt, **gen_args)
                    else:
                        cot = await self.llm.generate_async(prompt, **gen_args)
                    return idx, q, cot
                except Exception as e:
                    logger.warning("Async retry %d for candidate demo '%s': %s", attempt + 1, q, e)
                    if attempt < self.max_retries:
                        await asyncio.sleep(0.5 * (2**attempt))
            logger.error(
                "Failed to generate async demo for '%s' after %d retries",
                q,
                self.max_retries + 1,
            )
            return idx, q, None

        tasks = [generate_demo(idx, q) for idx, q in candidate_demos_info]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        demos: List[str] = []
        for res in results:
            if isinstance(res, Exception):
                logger.error("Async demo generation failed: %s", res, exc_info=False)
                continue
            idx, q, cot = res
            if cot is not None and count_steps(cot) <= self.max_steps:
                demos.append(f"Q: {q}\nA: {cot}")

        if len(demos) < self.n_demos:
            logger.warning(
                "Could only build %d demos async; need %d. Proceeding with available demos.",
                len(demos),
                self.n_demos,
            )
        if not demos:
            raise RuntimeError("Failed to build any valid demos asynchronously.")

        self.demos = demos

    def run(self, test_q: str, **kwargs: Any) -> str:
        """Runs the Auto-CoT strategy for a given test question.

        Constructs a prompt using the generated demonstrations and the test question,
        then calls the LLM to generate the final answer.

        Args:
            test_q: The test question to answer.
            **kwargs: Additional arguments passed to the LLM generation call.

        Returns:
            The LLM-generated answer string.

        Raises:
            RuntimeError: If `fit` or `fit_async` has not been called successfully first.
        """
        if self.demos is None:
            raise RuntimeError("Call fit() or fit_async() before run()")
        context = "\n\n".join(self.demos)
        payload = f"{context}\n\nQ: {test_q}\nA: {self.prompt_template}"
        logger.debug("AutoCoT payload:\n%s", payload)
        return self.llm.generate(
            payload,
            max_tokens=kwargs.pop("max_tokens", self.max_tokens),
            seed=kwargs.pop("seed", self.rand_seed),
            **kwargs,
        )

    async def run_async(self, test_q: str, **kwargs: Any) -> str:
        """Asynchronously runs the Auto-CoT strategy for a given test question.

        Constructs a prompt using the generated demonstrations and the test question,
        then calls the LLM asynchronously to generate the final answer.

        Args:
            test_q: The test question to answer.
            **kwargs: Additional arguments passed to the async LLM generation call.

        Returns:
            The LLM-generated answer string.

        Raises:
            RuntimeError: If `fit` or `fit_async` has not been called successfully first.
        """
        if self.demos is None:
            raise RuntimeError("Call fit() or fit_async() before run_async()")
        context = "\n\n".join(self.demos)
        payload = f"{context}\n\nQ: {test_q}\nA: {self.prompt_template}"
        logger.debug("Async AutoCoT payload:\n%s", payload)
        return await self.llm.generate_async(
            payload,
            max_tokens=kwargs.pop("max_tokens", self.max_tokens),
            seed=kwargs.pop("seed", self.rand_seed),
            **kwargs,
        )
