"""Implements the Cluster-Dependent Weighted Chain-of-Thought (CDW-CoT) strategy."""

import asyncio
import logging
import time
from typing import Any, List, Optional, Tuple

import numpy as np

from ..clustering import BaseClusterer, KMeansClusterer
from ..embedding import BaseEmbedder, SentenceTransformerEmbedder
from ..model import BaseLLM
from ..utils import accuracy, exact_match

logger = logging.getLogger(__name__)


class CDWCoT:
    """Implements Cluster-Dependent Weighted Chain-of-Thought (CDW-CoT).

    CDW-CoT involves creating a pool of CoT demonstrations, clustering the
    training questions, and learning separate prompt selection distributions
    for each cluster. At inference time, the test question is assigned to the
    nearest cluster, and its corresponding distribution is used to sample
    prompts for the final context.

    Reference:
        Fang et al. (2025). "CDW-CoT: Clustered Distance-Weighted Chain-of-Thoughts Reasoning",
        https://arxiv.org/abs/2501.12226
    """

    def __init__(
        self,
        llm: BaseLLM,
        pool_size: int = 40,
        n_clusters: int = 8,
        lr: float = 0.1,
        sample_size: int = 5,
        *,
        seed: Optional[int] = None,
        max_tokens: Optional[int] = None,
        max_grad_norm: float = 1.0,
        init_pool_retries: int = 1,
        embedder: Optional[BaseEmbedder] = None,
        clusterer: Optional[BaseClusterer] = None,
    ) -> None:
        """Initializes the CDWCoT strategy handler.

        Args:
            llm: The language model instance.
            pool_size: The target size of the demonstration prompt pool.
            n_clusters: The number of clusters to divide the training questions into.
            lr: Learning rate for updating the cluster distributions during training.
            sample_size: Number of demonstrations to sample for context during training
                         validation and final inference. Also used as the batch size during training.
            seed: Random seed for clustering, sampling, and potential LLM calls.
            max_tokens: Default maximum tokens for LLM generation calls.
            max_grad_norm: Maximum norm for clipping gradients during training.
            init_pool_retries: Number of retries if LLM fails during initial pool generation.
            embedder: Embedding model instance. Defaults to SentenceTransformerEmbedder.
            clusterer: Clustering algorithm instance. Defaults to KMeansClusterer.
        """
        self.llm = llm
        self.pool_size = pool_size
        self.n_clusters = n_clusters
        self.lr = lr
        self.sample_size = sample_size
        self.seed = seed
        self.max_tokens = max_tokens
        self.max_grad_norm = max_grad_norm
        self.init_pool_retries = init_pool_retries
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.clusterer = clusterer or KMeansClusterer()

        self.cluster_centers: Optional[np.ndarray] = None
        self.PC: List[str] = []
        self.p_cluster: List[np.ndarray] = []
        self.pool_map: List[Tuple[int, str]] = []
        self.train_questions: List[str] = []
        self.train_answers: List[str] = []
        self.train_labels: List[int] = []

    def _is_valid_distribution(self, p: np.ndarray) -> bool:
        """Checks if a NumPy array represents a valid probability distribution."""
        return bool(p.size and np.all(p >= 0) and np.isclose(p.sum(), 1.0))

    def _select_pool_indices(self, questions: List[str]) -> List[Tuple[int, str]]:
        """Selects candidate questions for the prompt pool based on clustering.

        Embeds questions, clusters them, and selects candidates closest to each
        cluster centroid, aiming for a total pool size close to `pool_size`.

        Args:
            questions: The list of questions to select from.

        Returns:
            A list of tuples, each containing the original index and the text
            of a selected candidate question.

        Raises:
            ValueError: If `n_clusters` is <= 0.
            RuntimeError: If embedding fails.
        """
        N = len(questions)
        effective_n = min(self.n_clusters, N)
        if effective_n <= 0:
            raise ValueError("Cannot initialize pool with zero clusters")

        logger.info(f"Encoding {N} questions for clustering...")
        embs_list = self.embedder.encode(questions)
        if len(embs_list) == 0:
            raise RuntimeError("Embedding failed to produce results.")
        embs = np.stack(embs_list)

        logger.info(f"Clustering embeddings into {effective_n} clusters...")
        labels, centers = self.clusterer.cluster(embs, effective_n, random_seed=self.seed or 0)
        self.cluster_centers = centers
        self.train_labels = labels.tolist()

        m: dict[int, str] = {}
        for c in range(effective_n):
            idxs = [i for i, lab in enumerate(labels) if lab == c]
            if not idxs:
                logger.debug(f"Cluster {c} has no associated questions.")
                continue

            k = (
                min(len(idxs), max(1, round(len(idxs) / N * self.pool_size)))
                if self.pool_size > 0
                else 0
            )
            logger.debug(f"Cluster {c} (size {len(idxs)}) sampling k={k} candidates for pool.")
            if k <= 0:
                continue

            d = np.linalg.norm(embs[idxs] - centers[c], axis=1)
            sorted_indices_in_cluster = np.argsort(d)
            for i in sorted_indices_in_cluster[:k]:
                original_index = idxs[i]
                m.setdefault(original_index, questions[original_index])

        selected_candidates = sorted(m.items())
        logger.info(
            f"Selected {len(selected_candidates)} unique pool candidate questions across clusters."
        )
        return selected_candidates

    def init_pool(self, questions: List[str], answers: List[str], **kwargs: Any) -> None:
        """Initializes the prompt pool by generating CoT for selected candidates.

        Uses `_select_pool_indices` to get candidates, then generates CoT reasoning
        for each using the LLM. Stores successful generations in `self.PC`. Also
        initializes uniform probability distributions (`self.p_cluster`) for each cluster.

        Args:
            questions: List of training questions.
            answers: Corresponding list of training answers.
            **kwargs: Additional arguments passed to the LLM generation call.

        Raises:
            ValueError: If question and answer lists have different lengths.
            RuntimeError: If pool selection yields no candidates or if all CoT
                          generations fail.
        """
        if len(questions) != len(answers):
            raise ValueError("Questions and answers list length mismatch.")

        self.train_questions = questions
        self.train_answers = answers
        pool_candidates = self._select_pool_indices(questions)

        if not pool_candidates:
            raise RuntimeError(
                "Prompt pool selection resulted in zero candidates. Check data or parameters."
            )

        logger.info(f"Generating initial CoT prompts for {len(pool_candidates)} candidates...")
        cots: dict[int, str] = {}
        successful_indices: List[int] = []
        failed_indices: List[int] = []

        for idx, q in pool_candidates:
            prompt = f"Q: {q}\nA: Let's think step by step."
            cot: Optional[str] = None
            for attempt in range(self.init_pool_retries + 1):
                try:
                    gen_seed = (
                        (self.seed + idx * (self.init_pool_retries + 1) + attempt)
                        if self.seed is not None
                        else None
                    )
                    cot = self.llm.generate(
                        prompt,
                        max_tokens=kwargs.pop("max_tokens", self.max_tokens),
                        seed=gen_seed,
                        **kwargs,
                    )
                    cots[idx] = f"Q: {q}\nA: {cot}"
                    successful_indices.append(idx)
                    logger.debug(f"Successfully generated CoT for pool candidate index {idx}.")
                    break
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed for pool index {idx}: {e}")
                    if attempt < self.init_pool_retries:
                        time.sleep(0.5 * 2**attempt)
                    else:
                        logger.error(
                            "Failed to generate CoT for pool index %d ('%s') after %d retries: %s",
                            idx,
                            q[:50] + "...",
                            self.init_pool_retries + 1,
                            e,
                        )
                        failed_indices.append(idx)

        self.PC = [cots[idx] for idx, _ in pool_candidates if idx in successful_indices]
        self.pool_map = [(idx, q) for idx, q in pool_candidates if idx in successful_indices]
        M = len(self.PC)

        if M == 0:
            raise RuntimeError("Prompt pool is empty after init_pool - all CoT generations failed.")
        elif failed_indices:
            logger.warning(f"Failed to generate CoT for {len(failed_indices)} pool candidates.")

        num_cl = self.cluster_centers.shape[0] if self.cluster_centers is not None else 0
        self.p_cluster = [np.ones(M) / M for _ in range(num_cl)]
        logger.info(
            f"Initialized prompt pool with {M} prompts and {num_cl} uniform cluster distributions."
        )

    async def init_pool_async(
        self,
        questions: List[str],
        answers: List[str],
        semaphore: Optional[asyncio.Semaphore] = None,
        **kwargs: Any,
    ) -> None:
        """Asynchronously initializes the prompt pool.

        Similar to `init_pool` but performs LLM generations asynchronously.

        Args:
            questions: List of training questions.
            answers: Corresponding list of training answers.
            semaphore: Optional asyncio.Semaphore to limit concurrent LLM calls.
            **kwargs: Additional arguments passed to the async LLM generation call.

        Raises:
            ValueError: If question and answer lists have different lengths.
            RuntimeError: If pool selection yields no candidates or if all CoT
                          generations fail.
        """
        if len(questions) != len(answers):
            raise ValueError("Questions and answers list length mismatch.")

        self.train_questions = questions
        self.train_answers = answers
        pool_candidates = self._select_pool_indices(questions)

        if not pool_candidates:
            raise RuntimeError(
                "Prompt pool selection resulted in zero candidates. Check data or parameters."
            )

        logger.info(
            f"Generating initial CoT prompts asynchronously for {len(pool_candidates)} candidates..."
        )

        async def gen(idx: int, q: str) -> None | tuple[int, str] | tuple[int, None]:
            prompt = f"Q: {q}\nA: Let's think step by step."
            for attempt in range(self.init_pool_retries + 1):
                try:
                    gen_seed = (
                        (self.seed + idx * (self.init_pool_retries + 1) + attempt)
                        if self.seed is not None
                        else None
                    )
                    local_kwargs = kwargs.copy()
                    gen_args = {
                        "max_tokens": local_kwargs.pop("max_tokens", self.max_tokens),
                        "seed": gen_seed,
                        **local_kwargs,
                    }
                    if semaphore:
                        async with semaphore:
                            cot = await self.llm.generate_async(prompt, **gen_args)
                    else:
                        cot = await self.llm.generate_async(prompt, **gen_args)
                    logger.debug(
                        f"Successfully generated async CoT for pool candidate index {idx}."
                    )
                    return idx, cot
                except Exception as e:
                    logger.warning(f"Async attempt {attempt + 1} failed for pool index {idx}: {e}")
                    if attempt < self.init_pool_retries:
                        await asyncio.sleep(0.5 * 2**attempt)
                    else:
                        logger.error(
                            "Failed async CoT gen for pool index %d ('%s') after %d retries: %s",
                            idx,
                            q[:50] + "...",
                            self.init_pool_retries + 1,
                            e,
                        )
                        return idx, None

        tasks = [gen(idx, q) for idx, q in pool_candidates]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        cots: dict[int, str] = {}
        successful_indices: List[int] = []
        failed_indices: List[int] = []
        for i, res in enumerate(results):
            original_index = pool_candidates[i][0]
            if isinstance(res, Exception):
                logger.error(f"Async generation task failed for index {original_index}: {res}")
                failed_indices.append(original_index)
            elif isinstance(res, tuple) and len(res) == 2:
                idx, cot_result = res
                if cot_result is not None:
                    cots[idx] = f"Q: {self.train_questions[idx]}\nA: {cot_result}"
                    successful_indices.append(idx)
                else:
                    failed_indices.append(idx)
            else:
                logger.error(
                    f"Unexpected result type from async generation task for index {original_index}: {type(res)}"
                )
                failed_indices.append(original_index)

        self.PC = [cots[idx] for idx in successful_indices]
        self.pool_map = [(idx, self.train_questions[idx]) for idx in successful_indices]
        M = len(self.PC)

        if M == 0:
            raise RuntimeError(
                "Prompt pool empty after async init_pool - all CoT generations failed."
            )
        elif failed_indices:
            logger.warning(
                f"Failed to generate async CoT for {len(failed_indices)} pool candidates."
            )

        num_cl = self.cluster_centers.shape[0] if self.cluster_centers is not None else 0
        self.p_cluster = [np.ones(M) / M for _ in range(num_cl)]
        logger.info(
            f"Initialized prompt pool asynchronously with {M} prompts and {num_cl} uniform cluster distributions."
        )

    def train(
        self, val_split: float = 0.2, epochs: int = 100, patience: int = 5, **kwargs: Any
    ) -> None:
        """Trains the cluster-dependent prompt distributions.

        Iterates through epochs, sampling batches from each cluster's training data.
        For each sample, it selects a prompt from the pool based on the current
        cluster distribution, generates an answer, calculates a loss (0 for correct,
        1 for incorrect), computes policy gradients, and updates the distribution.
        Uses a validation set for early stopping.

        Args:
            val_split: Fraction of data within each cluster to use for validation.
            epochs: Maximum number of training epochs.
            patience: Number of epochs with no improvement on validation accuracy
                      before early stopping.
            **kwargs: Additional arguments passed to LLM generation calls during
                      training and validation.

        Raises:
            RuntimeError: If `init_pool` has not been called successfully first.
        """
        if not self.PC or self.cluster_centers is None or not self.p_cluster:
            raise RuntimeError("Call init_pool first before training.")

        logger.info("Starting synchronous training...")
        rnd = np.random.RandomState(self.seed) if self.seed is not None else np.random.RandomState()
        M = len(self.PC)
        nc = len(self.p_cluster)

        cluster_idxs = {
            c: [i for i, lab in enumerate(self.train_labels) if lab == c] for c in range(nc)
        }

        for c, idxs in cluster_idxs.items():
            if not idxs:
                logger.debug(f"Skipping training for empty cluster {c}.")
                if c < len(self.p_cluster) and not self._is_valid_distribution(self.p_cluster[c]):
                    self.p_cluster[c] = np.ones(M) / M
                continue

            rnd.shuffle(idxs)
            split_idx = max(1, int(len(idxs) * (1 - val_split)))
            train_idx, val_idx = idxs[:split_idx], idxs[split_idx:]

            if not val_idx:
                logger.warning(
                    "Validation set empty for cluster %d (size %d, split %f). Using training set for validation.",
                    c,
                    len(idxs),
                    val_split,
                )
                val_idx = train_idx
            if not train_idx:
                logger.warning(
                    f"Training set empty for cluster {c}. Skipping training for this cluster."
                )
                continue

            logger.info(
                f"Training cluster {c}: {len(train_idx)} train samples, {len(val_idx)} validation samples."
            )

            p = self.p_cluster[c].copy()
            if not self._is_valid_distribution(p):
                logger.warning(
                    f"Initial distribution for cluster {c} invalid, resetting to uniform."
                )
                p = np.ones(M) / M

            best_p = p.copy()
            best_acc = -1.0
            no_imp = 0

            for epoch in range(epochs):
                batch = rnd.choice(
                    train_idx, size=min(len(train_idx), self.sample_size), replace=False
                )

                losses: List[float] = []
                grads = np.zeros_like(p)
                batch_results: List[Tuple[int, float]] = []

                for j, orig_idx in enumerate(batch):
                    m = rnd.choice(M, p=p)
                    q = self.train_questions[orig_idx]
                    prev = self.PC[m]
                    payload = f"{prev}\n\nQ: {q}\nA: Let's think step by step."
                    loss = 1.0

                    try:
                        gen_seed = (self.seed or 0) + epoch * len(batch) + j
                        local_kwargs = kwargs.copy()
                        out = self.llm.generate(
                            payload,
                            max_tokens=local_kwargs.pop("max_tokens", self.max_tokens),
                            seed=gen_seed,
                            **local_kwargs,
                        )
                        if exact_match(out, self.train_answers[orig_idx]):
                            loss = 0.0
                    except Exception as e:
                        logger.error(
                            f"Sync train generation failed for q_idx {orig_idx}, p_idx {m}: {e}"
                        )

                    batch_results.append((m, loss))
                    losses.append(loss)

                if not losses:
                    continue

                mean_loss = np.mean(losses)
                for m_idx, loss in batch_results:
                    advantage = loss - mean_loss
                    grads[m_idx] += -advantage / max(p[m_idx], 1e-9)

                norm = np.linalg.norm(grads)
                if norm > self.max_grad_norm:
                    grads *= self.max_grad_norm / norm

                p = p - self.lr * (grads / len(losses))
                p = np.clip(p, 1e-9, None)

                p_sum = p.sum()
                p = p / p_sum if p_sum > 1e-9 else np.ones_like(p) / p.size

                val_preds = []
                for val_orig in val_idx:
                    top_indices = np.argsort(-p)[: min(self.sample_size, M)]
                    ctx = "\n\n".join(self.PC[i] for i in top_indices)
                    vp = f"{ctx}\n\nQ: {self.train_questions[val_orig]}\nA: Let's think step by step."
                    val_out = ""
                    try:
                        val_seed = (self.seed or 0) + val_orig
                        local_kwargs_val = kwargs.copy()
                        val_out = self.llm.generate(
                            vp,
                            max_tokens=local_kwargs_val.pop("max_tokens", self.max_tokens),
                            seed=val_seed,
                            **local_kwargs_val,
                        )
                    except Exception as e:
                        logger.error(f"Sync validation generation failed for q_idx {val_orig}: {e}")
                        val_out = "[ERROR]"
                    val_preds.append(val_out)

                val_golds = [self.train_answers[i] for i in val_idx]
                acc = accuracy(val_preds, val_golds)
                logger.debug(
                    f"Cluster {c} Epoch {epoch + 1}: Train Loss={mean_loss:.3f}, Val Acc={acc:.3f}"
                )

                if acc > best_acc:
                    best_acc, best_p, no_imp = acc, p.copy(), 0
                else:
                    no_imp += 1
                    if no_imp >= patience:
                        logger.info(
                            f"Early stopping for cluster {c} at epoch {epoch + 1} (Val Acc: {best_acc:.3f})"
                        )
                        break

            self.p_cluster[c] = best_p
            logger.info(f"Finished training cluster {c}. Best Val Acc: {best_acc:.3f}")

    async def train_async(
        self,
        val_split: float = 0.2,
        epochs: int = 100,
        patience: int = 5,
        semaphore: Optional[asyncio.Semaphore] = None,
        **kwargs: Any,
    ) -> None:
        """Asynchronously trains the cluster-dependent prompt distributions.

        Similar to `train`, but performs LLM generation calls asynchronously.

        Args:
            val_split: Fraction of data within each cluster for validation.
            epochs: Maximum number of training epochs.
            patience: Epochs with no validation improvement before early stopping.
            semaphore: Optional asyncio.Semaphore to limit concurrent LLM calls.
            **kwargs: Additional arguments for async LLM calls during training/validation.

        Raises:
            RuntimeError: If `init_pool` or `init_pool_async` has not been called first.
        """
        if not self.PC or self.cluster_centers is None or not self.p_cluster:
            raise RuntimeError("Call init_pool_async first before training.")

        logger.info("Starting asynchronous training...")
        rnd = np.random.RandomState(self.seed) if self.seed is not None else np.random.RandomState()
        M = len(self.PC)
        nc = len(self.p_cluster)

        cluster_idxs = {
            c: [i for i, lab in enumerate(self.train_labels) if lab == c] for c in range(nc)
        }

        training_coroutines = []

        for c, idxs in cluster_idxs.items():
            if not idxs:
                logger.debug(f"Skipping async training for empty cluster {c}.")
                if c < len(self.p_cluster) and not self._is_valid_distribution(self.p_cluster[c]):
                    self.p_cluster[c] = np.ones(M) / M
                continue

            rnd.shuffle(idxs)
            split_idx = max(1, int(len(idxs) * (1 - val_split)))
            train_idx, val_idx = idxs[:split_idx], idxs[split_idx:]
            if not val_idx:
                logger.warning(
                    "Async Validation set empty for cluster %d. Using training set for validation.",
                    c,
                )
                val_idx = train_idx
            if not train_idx:
                logger.warning(f"Async Training set empty for cluster {c}. Skipping training.")
                continue

            logger.info(
                f"Starting async training for cluster {c}: {len(train_idx)} train, {len(val_idx)} val."
            )

            async def train_cluster(
                cluster_index: int,
                initial_p: np.ndarray,
                train_indices: List[int],
                val_indices: List[int],
            ) -> None:
                p = initial_p.copy()
                if not self._is_valid_distribution(p):
                    logger.warning(
                        f"Async initial dist for cluster {cluster_index} invalid, resetting."
                    )
                    p = np.ones(M) / M

                best_p = p.copy()
                best_acc = -1.0
                no_imp = 0

                for epoch in range(epochs):
                    batch_indices = rnd.choice(
                        train_indices, size=min(len(train_indices), self.sample_size), replace=False
                    )

                    async def process_batch_item(j: int, orig_idx: int) -> Tuple[int, float]:
                        m = rnd.choice(M, p=p)
                        q = self.train_questions[orig_idx]
                        prev = self.PC[m]
                        payload = f"{prev}\n\nQ: {q}\nA: Let's think step by step."
                        loss = 1.0
                        try:
                            gen_seed = (self.seed or 0) + epoch * len(batch_indices) + j
                            local_kwargs = kwargs.copy()
                            gen_args = {
                                "max_tokens": local_kwargs.pop("max_tokens", self.max_tokens),
                                "seed": gen_seed,
                                **local_kwargs,
                            }
                            if semaphore:
                                async with semaphore:
                                    out = await self.llm.generate_async(payload, **gen_args)
                            else:
                                out = await self.llm.generate_async(payload, **gen_args)

                            if exact_match(out, self.train_answers[orig_idx]):
                                loss = 0.0
                        except Exception as e:
                            logger.error(
                                f"Async train generation failed q_idx {orig_idx}, p_idx {m}: {e}"
                            )
                        return m, loss

                    batch_results_tuples: List[Tuple[int, float]] = await asyncio.gather(
                        *(
                            process_batch_item(j, orig_idx)
                            for j, orig_idx in enumerate(batch_indices)
                        )
                    )

                    losses = [loss for _, loss in batch_results_tuples]
                    if not losses:
                        continue

                    mean_loss = np.mean(losses)
                    grads = np.zeros_like(p)
                    for m_idx, loss in batch_results_tuples:
                        advantage = loss - mean_loss
                        grads[m_idx] += -advantage / max(p[m_idx], 1e-9)

                    norm = np.linalg.norm(grads)
                    if norm > self.max_grad_norm:
                        grads *= self.max_grad_norm / norm

                    p = p - self.lr * (grads / len(losses))
                    p = np.clip(p, 1e-9, None)
                    p_sum = p.sum()
                    p = p / p_sum if p_sum > 1e-9 else np.ones_like(p) / p.size

                    async def validate_item(val_orig_idx: int) -> str:
                        top_indices = np.argsort(-p)[: min(self.sample_size, M)]
                        ctx = "\n\n".join(self.PC[i] for i in top_indices)
                        vp = f"{ctx}\n\nQ: {self.train_questions[val_orig_idx]}\nA: Let's think step by step."
                        val_out = "[ERROR]"
                        try:
                            val_seed = (self.seed or 0) + val_orig_idx
                            local_kwargs_val = kwargs.copy()
                            val_gen_args = {
                                "max_tokens": local_kwargs_val.pop("max_tokens", self.max_tokens),
                                "seed": val_seed,
                                **local_kwargs_val,
                            }
                            if semaphore:
                                async with semaphore:
                                    val_out = await self.llm.generate_async(vp, **val_gen_args)
                            else:
                                val_out = await self.llm.generate_async(vp, **val_gen_args)
                        except Exception as e:
                            logger.error(
                                f"Async validation generation failed q_idx {val_orig_idx}: {e}"
                            )
                        return val_out

                    val_preds = await asyncio.gather(
                        *(validate_item(val_idx) for val_idx in val_indices)
                    )

                    val_golds = [self.train_answers[i] for i in val_indices]
                    acc = accuracy(val_preds, val_golds)
                    logger.debug(
                        f"Async Cluster {cluster_index} Epoch {epoch + 1}: Train Loss={mean_loss:.3f}, Val Acc={acc:.3f}"
                    )

                    if acc > best_acc:
                        best_acc, best_p, no_imp = acc, p.copy(), 0
                    else:
                        no_imp += 1
                        if no_imp >= patience:
                            logger.info(
                                f"Async early stopping for cluster {cluster_index} at epoch {epoch + 1} (Val Acc: {best_acc:.3f})"
                            )
                            break

                self.p_cluster[cluster_index] = best_p
                logger.info(
                    f"Finished async training cluster {cluster_index}. Best Val Acc: {best_acc:.3f}"
                )

            training_coroutines.append(train_cluster(c, self.p_cluster[c], train_idx, val_idx))

        await asyncio.gather(*training_coroutines)
        logger.info("Asynchronous CDW-CoT training complete for all clusters.")

    def _calculate_combined_distribution(
        self, question: str, temperature: float = 0.3
    ) -> np.ndarray:
        """Calculates the distance-weighted prompt selection distribution for a question.

        This method implements the Distance-Weighting (Dist-W) approach described
        in the paper. It embeds the input question, calculates its Euclidean distance
        to all learned cluster centers, computes softmax weights based on these distances and
        a temperature parameter, and finally returns a combined prompt probability distribution
        by taking a weighted average of the optimal distributions learned for each cluster.

        Args:
            question: The input test question string.
            temperature: The temperature parameter for the softmax weight calculation.
                         Controls sensitivity to distance.
                         Defaults to 0.3 based on the paper's findings.

        Returns:
            A NumPy array representing the combined probability distribution over the
            prompt pool for the given question. Falls back to a uniform
            distribution if prerequisites (pool, centers, cluster distributions)
            are missing or if errors occur during calculation.

        Raises:
            RuntimeError: If the prompt pool is empty.
            ValueError: If the question embedding dimension does not match the
                        cluster center dimensions after reshaping.
        """
        M = len(self.PC)
        if M == 0:
            raise RuntimeError("Prompt pool is empty. Cannot calculate distribution.")

        if (
            self.cluster_centers is None
            or not self.p_cluster
            or len(self.p_cluster) != self.cluster_centers.shape[0]
        ):
            logger.warning(
                "Cluster centers or probabilities not initialized correctly. Falling back to uniform distribution."
            )
            return np.ones(M) / M

        try:
            logger.debug(f"Encoding question for distribution calculation: '{question[:50]}...'")
            q_emb_list = self.embedder.encode([question])

            if len(q_emb_list) == 0 or q_emb_list[0] is None:
                raise ValueError("Encoding failed or returned None for the input question.")

            q_emb = np.stack(q_emb_list)[0]

            if q_emb is not None and self.cluster_centers.size > 0:
                if q_emb.shape != self.cluster_centers.shape[1:]:
                    q_emb_reshaped = q_emb.reshape(1, -1)
                    if q_emb_reshaped.shape[1] != self.cluster_centers.shape[1]:
                        raise ValueError(
                            f"Question embedding dimension {q_emb_reshaped.shape} doesn't match cluster center dimension {self.cluster_centers.shape}"
                        )
                    q_emb_final = q_emb_reshaped
                else:
                    q_emb_final = q_emb

                distances = np.linalg.norm(self.cluster_centers - q_emb_final, axis=1)

                temp_safe = max(temperature, 1e-6)

                neg_dist_over_temp = -distances / temp_safe
                neg_dist_over_temp = neg_dist_over_temp - np.max(neg_dist_over_temp)
                exp_values = np.exp(neg_dist_over_temp)
                weights = exp_values / np.sum(exp_values)

                logger.debug(f"Calculated weights for clusters: {weights}")

                final_distribution = np.zeros(M, dtype=float)
                valid_distributions_found = False
                for i, weight in enumerate(weights):
                    if i < len(self.p_cluster) and self._is_valid_distribution(self.p_cluster[i]):
                        final_distribution += weight * self.p_cluster[i]
                        valid_distributions_found = True
                    else:
                        logger.warning(f"Skipping invalid or missing distribution for cluster {i}")

                if not valid_distributions_found:
                    logger.warning(
                        "No valid cluster distributions found for weighting. Falling back to uniform."
                    )
                    return np.ones(M) / M

                final_sum = final_distribution.sum()
                if final_sum > 1e-9:
                    final_distribution /= final_sum
                else:
                    logger.warning(
                        "Weighted distribution sum is near zero. Falling back to uniform."
                    )
                    return np.ones(M) / M

                logger.debug("Using distance-weighted combined distribution.")
                return final_distribution
            else:
                logger.warning(
                    "Could not use question embedding or no cluster centers available. Falling back to uniform."
                )

        except Exception as e:
            logger.error(
                "Error calculating distribution for question '%s': %s. Falling back to uniform.",
                question[:50] + "...",
                e,
                exc_info=True,
            )

        logger.debug("Falling back to uniform distribution.")
        return np.ones(M) / M

    def run(self, test_q: str, **kwargs: Any) -> str:
        """Runs the CDW-CoT strategy for a given test question.

        Determines the appropriate prompt distribution using `_calculate_combined_distribution`,
        samples top prompts according to this distribution, constructs the context,
        and generates the final answer using the LLM.

        Args:
            test_q: The test question to answer.
            **kwargs: Additional arguments passed to the final LLM generation call.

        Returns:
            The LLM-generated answer string.

        Raises:
            RuntimeError: If the prompt pool is empty.
        """
        dist = self._calculate_combined_distribution(test_q)
        M = len(self.PC)
        if M == 0:
            raise RuntimeError("Prompt pool is empty, cannot run.")

        top_indices = np.argsort(-dist)[: min(self.sample_size, M)]
        logger.debug(
            f"Selected top prompt indices for sync run: {top_indices} based on distribution."
        )
        ctxt = "\n\n".join(self.PC[i] for i in top_indices)
        payload = f"{ctxt}\n\nQ: {test_q}\nA: Let's think step by step."

        gen_seed = (self.seed + len(self.train_questions)) if self.seed is not None else None
        logger.info(f"Generating sync answer for: '{test_q[:50]}...'")
        return self.llm.generate(
            payload,
            max_tokens=kwargs.pop("max_tokens", self.max_tokens),
            seed=gen_seed,
            **kwargs,
        )

    async def run_async(
        self, test_q: str, semaphore: Optional[asyncio.Semaphore] = None, **kwargs: Any
    ) -> str:
        """Asynchronously runs the CDW-CoT strategy for a given test question.

        Similar to `run`, but performs the final LLM generation call asynchronously.

        Args:
            test_q: The test question to answer.
            semaphore: Optional asyncio.Semaphore to limit concurrent LLM calls.
            **kwargs: Additional arguments passed to the final async LLM generation call.

        Returns:
            The LLM-generated answer string.

        Raises:
            RuntimeError: If the prompt pool is empty.
        """
        dist = self._calculate_combined_distribution(test_q)
        M = len(self.PC)
        if M == 0:
            raise RuntimeError("Prompt pool is empty, cannot run.")

        top_indices = np.argsort(-dist)[: min(self.sample_size, M)]
        logger.debug(
            f"Selected top prompt indices for async run: {top_indices} based on distribution."
        )
        ctxt = "\n\n".join(self.PC[i] for i in top_indices)
        payload = f"{ctxt}\n\nQ: {test_q}\nA: Let's think step by step."

        gen_seed = (self.seed + len(self.train_questions)) if self.seed is not None else None
        logger.info(f"Generating async answer for: '{test_q[:50]}...'")

        local_kwargs = kwargs.copy()
        gen_args = {
            "max_tokens": local_kwargs.pop("max_tokens", self.max_tokens),
            "seed": gen_seed,
            **local_kwargs,
        }

        if semaphore:
            async with semaphore:
                return await self.llm.generate_async(payload, **gen_args)
        else:
            return await self.llm.generate_async(payload, **gen_args)
