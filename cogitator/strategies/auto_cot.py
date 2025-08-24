import asyncio
import logging
import time
from typing import Any, List, Optional, Tuple

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
        Zhang et al. (2022) "Automatic Chain of Thought Prompting in Large Language Models".
        https://arxiv.org/abs/2210.03493
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
            rand_seed: Base random seed for clustering and LLM seeding. LLM calls will
                       use variations of this seed.
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
        representatives, generating CoT reasoning for them using varied seeds,
        and filtering based on length and step count criteria.

        Args:
            questions: A list of questions to build demonstrations from.

        Raises:
            ValueError: If the number of questions is lower than `n_demos`.
            RuntimeError: If embedding or clustering fails, or if no valid demos
                can be generated.
        """
        if len(questions) < self.n_demos:
            raise ValueError(f"Need >= {self.n_demos} questions, got {len(questions)}")

        logger.info("Encoding questions for AutoCoT fitting...")
        embs_list = self.embedder.encode(questions)
        if len(embs_list) == 0:
            raise RuntimeError("Embedding failed to produce results.")
        embs = np.stack(embs_list)

        logger.info("Clustering questions...")
        labels, centers = self.clusterer.cluster(
            embs, self.n_demos, random_seed=self.rand_seed or 0
        )

        logger.info("Selecting candidate demonstrations...")
        candidate_demos: List[Tuple[int, str]] = []
        for c in range(self.n_demos):
            idxs_in_cluster = np.where(labels == c)[0]
            if idxs_in_cluster.size == 0:
                logger.debug(f"Cluster {c} is empty, skipping.")
                continue

            # Calculate distances within the cluster
            cluster_embs = embs[idxs_in_cluster]
            dists = np.linalg.norm(cluster_embs - centers[c], axis=1)
            sorted_relative_indices = np.argsort(dists)

            # Iterate through questions closest to the centroid first
            found_candidate = False
            for relative_idx in sorted_relative_indices:
                original_idx = idxs_in_cluster[relative_idx]
                q = questions[original_idx]
                if approx_token_length(q) <= self.max_q_tokens:
                    candidate_demos.append((original_idx, q))
                    logger.debug(f"Selected candidate from cluster {c}: Q index {original_idx}")
                    found_candidate = True
                    break  # Take only the first valid one closest to centroid

            if not found_candidate:
                logger.debug(f"No suitable candidate found for cluster {c} within token limits.")

        logger.info(f"Generating CoT reasoning for {len(candidate_demos)} candidates...")
        demos: List[str] = []
        # Use enumerate for a simple loop counter if needed, but idx is usually better
        for _demo_idx, (original_q_idx, q) in enumerate(candidate_demos):
            prompt = f"Q: {q}\nA: {self.prompt_template}"
            cot: Optional[str] = None
            for attempt in range(self.max_retries + 1):
                # --- Seed Refinement ---
                # Vary seed based on the original question index and attempt number
                iter_seed: Optional[int] = None
                if self.rand_seed is not None:
                    # Combine base seed, question index, and attempt number
                    # Multiplying attempt helps space out seeds more
                    iter_seed = self.rand_seed + original_q_idx + attempt * 101
                # --- End Seed Refinement ---

                try:
                    logger.debug(
                        f"Attempt {attempt + 1} for Q idx {original_q_idx} with seed {iter_seed}"
                    )
                    cot = self.llm.generate(
                        prompt,
                        max_tokens=self.max_tokens,
                        seed=iter_seed,  # Use the varied seed
                    )
                    break  # Success
                except Exception as e:
                    logger.warning(
                        f"Retry {attempt + 1}/{self.max_retries + 1} for demo Q idx {original_q_idx}: {e}",
                        exc_info=(logger.getEffectiveLevel() <= logging.DEBUG),
                        # Show traceback in debug
                    )
                    if attempt < self.max_retries:
                        # Optional: add a small delay before retrying
                        time.sleep(0.5 * (2**attempt))

            if cot is None:
                logger.error(
                    "Failed to generate demo for Q idx %d ('%s') after %d retries",
                    original_q_idx,
                    q[:50] + "...",
                    self.max_retries + 1,
                )
                continue  # Skip this candidate

            # Filter based on step count
            steps_count = count_steps(cot)
            if steps_count <= self.max_steps:
                demos.append(f"Q: {q}\nA: {cot}")
                logger.debug(
                    f"Successfully generated and filtered demo for Q idx {original_q_idx} ({steps_count} steps)"
                )
            else:
                logger.debug(
                    f"Generated demo for Q idx {original_q_idx} discarded ({steps_count} steps > max {self.max_steps})"
                )

        if len(demos) < self.n_demos:
            logger.warning(
                "Could only build %d final demos (needed %d). Proceeding with available demos.",
                len(demos),
                self.n_demos,
            )
        if not demos:
            logger.error("Failed to build any valid demos after generation and filtering.")
            raise RuntimeError("Failed to build any valid demos.")

        self.demos = demos
        logger.info(f"AutoCoT fitting complete. Generated {len(demos)} demonstrations.")

    async def fit_async(
        self, questions: List[str], semaphore: Optional[asyncio.Semaphore] = None
    ) -> None:
        """Asynchronously builds the demonstration pool using the Auto-CoT process.

        Similar to `fit`, but performs LLM generation calls asynchronously
        using varied seeds.

        Args:
            questions: A list of questions to build demonstrations from.
            semaphore: An optional asyncio.Semaphore to limit concurrent LLM calls.

        Raises:
            ValueError: If the number of questions is lower than `n_demos`.
            RuntimeError: If embedding or clustering fails, or if no valid demos
                can be generated.
        """
        if len(questions) < self.n_demos:
            raise ValueError(f"Need >= {self.n_demos} questions, got {len(questions)}")

        logger.info("Encoding questions for async AutoCoT fitting...")
        embs_list = self.embedder.encode(questions)
        if len(embs_list) == 0:
            raise RuntimeError("Embedding failed to produce results.")
        embs = np.stack(embs_list)

        logger.info("Clustering questions async...")
        labels, centers = self.clusterer.cluster(
            embs, self.n_demos, random_seed=self.rand_seed or 0
        )

        logger.info("Selecting candidate demonstrations async...")
        candidate_demos_info: List[Tuple[int, str]] = []
        for c in range(self.n_demos):
            idxs_in_cluster = np.where(labels == c)[0]
            if idxs_in_cluster.size == 0:
                continue
            cluster_embs = embs[idxs_in_cluster]
            dists = np.linalg.norm(cluster_embs - centers[c], axis=1)
            sorted_relative_indices = np.argsort(dists)
            for relative_idx in sorted_relative_indices:
                original_idx = idxs_in_cluster[relative_idx]
                q = questions[original_idx]
                if approx_token_length(q) <= self.max_q_tokens:
                    candidate_demos_info.append((original_idx, q))
                    break

        logger.info(f"Generating CoT reasoning async for {len(candidate_demos_info)} candidates...")

        async def generate_demo(idx: int, q: str) -> Tuple[int, str, Optional[str]]:
            prompt = f"Q: {q}\nA: {self.prompt_template}"
            for attempt in range(self.max_retries + 1):
                iter_seed: Optional[int] = None
                if self.rand_seed is not None:
                    iter_seed = self.rand_seed + idx + attempt * 101

                try:
                    logger.debug(
                        f"Async attempt {attempt + 1} for Q idx {idx} with seed {iter_seed}"
                    )
                    gen_args = {
                        "max_tokens": self.max_tokens,
                        "seed": iter_seed,
                    }

                    if semaphore:
                        async with semaphore:
                            cot = await self.llm.generate_async(prompt, **gen_args)
                    else:
                        cot = await self.llm.generate_async(prompt, **gen_args)
                    return idx, q, cot
                except Exception as e:
                    logger.warning(
                        f"Async retry {attempt + 1}/{self.max_retries + 1} for demo Q idx {idx}: {e}",
                        exc_info=(logger.getEffectiveLevel() <= logging.DEBUG),
                    )
                    if attempt < self.max_retries:
                        await asyncio.sleep(0.5 * (2**attempt))

            logger.error(
                "Failed to generate async demo for Q idx %d ('%s') after %d retries",
                idx,
                q[:50] + "...",
                self.max_retries + 1,
            )
            return idx, q, None  # Failed after retries

        tasks = [generate_demo(idx, q) for idx, q in candidate_demos_info]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        demos: List[str] = []
        successful_generations = 0
        for res in results:
            if isinstance(res, Exception):
                if not isinstance(res, asyncio.CancelledError):
                    logger.error(f"Async demo generation task failed: {res}", exc_info=True)
                continue

            if isinstance(res, tuple) and len(res) == 3:
                _idx, q, cot = res
                if cot is not None:
                    successful_generations += 1
                    steps_count = count_steps(cot)
                    if steps_count <= self.max_steps:
                        demos.append(f"Q: {q}\nA: {cot}")
                        logger.debug(
                            f"Successfully generated and filtered async demo for Q idx {_idx} ({steps_count} steps)"
                        )
                    else:
                        logger.debug(
                            f"Async demo for Q idx {_idx} discarded ({steps_count} steps > max {self.max_steps})"
                        )
            else:
                logger.error(f"Unexpected result type from gather: {type(res)} - {res}")

        if len(demos) < self.n_demos:
            logger.warning(
                "Could only build %d final demos async (needed %d). Proceeding with available demos.",
                len(demos),
                self.n_demos,
            )
        if not demos:
            logger.error(
                "Failed to build any valid demos asynchronously after generation and filtering."
            )
            raise RuntimeError("Failed to build any valid demos asynchronously.")

        self.demos = demos
        logger.info(f"Async AutoCoT fitting complete. Generated {len(demos)} demonstrations.")

    # Add type hint Any to **kwargs
    def run(self, test_q: str, **kwargs: Any) -> str:
        """Runs the Auto-CoT strategy for a given test question.

        Constructs a prompt using the generated demonstrations and the test question,
        then calls the LLM to generate the final answer. The base seed is used
        for this final generation unless overridden in kwargs.

        Args:
            test_q: The test question to answer.
            **kwargs: Additional arguments passed to the LLM generation call,
                      potentially overriding default seed, max_tokens, etc.

        Returns:
            The LLM-generated answer string.

        Raises:
            RuntimeError: If `fit` or `fit_async` has not been called successfully first.
        """
        if self.demos is None:
            raise RuntimeError("Call fit() or fit_async() before run()")

        context = "\n\n".join(self.demos)
        payload = f"{context}\n\nQ: {test_q}\nA: {self.prompt_template}"
        logger.debug(
            "AutoCoT final inference payload:\n%s", payload[:500] + "..."
        )  # Log truncated payload

        final_seed = kwargs.pop("seed", self.rand_seed)
        final_max_tokens = kwargs.pop("max_tokens", self.max_tokens)

        logger.info(f"Running final AutoCoT generation for question: '{test_q[:50]}...'")
        try:
            result = self.llm.generate(
                payload,
                max_tokens=final_max_tokens,
                seed=final_seed,
                **kwargs,
            )
            logger.info("Final generation successful.")
            return result
        except Exception as e:
            logger.error(f"Final AutoCoT generation failed: {e}", exc_info=True)
            # Depending on desired behavior, either raise e or return an error marker
            # raise e
            return "[ERROR: Final generation failed]"

    # Add type hint Any to **kwargs
    async def run_async(self, test_q: str, **kwargs: Any) -> str:
        """Asynchronously runs the Auto-CoT strategy for a given test question.

        Constructs a prompt using the generated demonstrations and the test question,
        then calls the LLM asynchronously to generate the final answer. The base
        seed is used for this final generation unless overridden in kwargs.

        Args:
            test_q: The test question to answer.
            **kwargs: Additional arguments passed to the async LLM generation call,
                      potentially overriding default seed, max_tokens, etc.

        Returns:
            The LLM-generated answer string.

        Raises:
            RuntimeError: If `fit` or `fit_async` has not been called successfully first.
        """
        if self.demos is None:
            raise RuntimeError("Call fit() or fit_async() before run_async()")

        context = "\n\n".join(self.demos)
        payload = f"{context}\n\nQ: {test_q}\nA: {self.prompt_template}"
        logger.debug("Async AutoCoT final inference payload:\n%s", payload[:500] + "...")

        final_seed = kwargs.pop("seed", self.rand_seed)
        final_max_tokens = kwargs.pop("max_tokens", self.max_tokens)

        logger.info(f"Running final async AutoCoT generation for question: '{test_q[:50]}...'")
        try:
            semaphore = kwargs.pop("semaphore", None)
            gen_args = {
                "max_tokens": final_max_tokens,
                "seed": final_seed,
                **kwargs,
            }
            if semaphore:
                async with semaphore:
                    result = await self.llm.generate_async(payload, **gen_args)
            else:
                result = await self.llm.generate_async(payload, **gen_args)

            logger.info("Final async generation successful.")
            return result
        except Exception as e:
            logger.error(f"Final async AutoCoT generation failed: {e}", exc_info=True)
            # raise e
            return "[ERROR: Final async generation failed]"
