import asyncio
import logging
from typing import List, Optional, Tuple

import numpy as np

from .model import BaseLLM
from .utils import approx_token_length, cluster_embeddings, count_steps, encode

logger = logging.getLogger(__name__)


class AutoCoT:
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
    ):
        self.llm = llm
        self.n_demos = n_demos
        self.max_q_tokens = max_q_tokens
        self.max_steps = max_steps
        self.prompt_template = prompt_template
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.rand_seed = rand_seed
        self.demos: Optional[List[str]] = None

    def fit(self, questions: List[str]) -> None:
        if len(questions) < self.n_demos:
            raise ValueError(f"Need >= {self.n_demos} questions, got {len(questions)}")

        embs = np.stack(encode(questions))
        labels, centers = cluster_embeddings(embs, self.n_demos, random_state=self.rand_seed or 0)

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
        if len(questions) < self.n_demos:
            raise ValueError(f"Need >= {self.n_demos} questions, got {len(questions)}")

        embs = np.stack(encode(questions))
        labels, centers = cluster_embeddings(embs, self.n_demos, random_state=self.rand_seed or 0)

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
                    if semaphore:
                        async with semaphore:
                            cot = await self.llm.generate_async(
                                prompt,
                                max_tokens=self.max_tokens,
                                seed=self.rand_seed,
                            )
                    else:
                        cot = await self.llm.generate_async(
                            prompt,
                            max_tokens=self.max_tokens,
                            seed=self.rand_seed,
                        )
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
            idx, q, cot = res  # type: ignore
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

    def answer(self, test_q: str) -> str:
        if self.demos is None:
            raise RuntimeError("Call fit(...) or fit_async(...) before answer(...)")
        context = "\n\n".join(self.demos)
        payload = f"{context}\n\nQ: {test_q}\nA: {self.prompt_template}"
        logger.debug("AutoCoT payload:\n%s", payload)
        return self.llm.generate(
            payload,
            max_tokens=self.max_tokens,
            seed=self.rand_seed,
        )

    async def answer_async(self, test_q: str) -> str:
        if self.demos is None:
            raise RuntimeError("Call fit(...) or fit_async(...) before answer_async(...)")
        context = "\n\n".join(self.demos)
        payload = f"{context}\n\nQ: {test_q}\nA: {self.prompt_template}"
        logger.debug("Async AutoCoT payload:\n%s", payload)
        return await self.llm.generate_async(
            payload,
            max_tokens=self.max_tokens,
            seed=self.rand_seed,
        )

    __call__ = answer
