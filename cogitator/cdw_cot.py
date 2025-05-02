import asyncio
import logging
import time
from typing import List, Optional, Tuple  # Added Any for Coroutine typing

import numpy as np

from .model import BaseLLM
from .utils import accuracy, cluster_embeddings, encode, exact_match

logger = logging.getLogger(__name__)


class CDWCoT:
    """
    Implements Clustered Distance-Weighted Chain-of-Thought (CDW-CoT).

    This method clusters training questions, generates CoT demonstrations
    for selected questions (prompt pool), learns a probability distribution
    over the prompt pool for each cluster, and uses these distributions
    to select relevant prompts for answering new questions.
    """

    def __init__(
        self,
        llm: BaseLLM,
        pool_size: int = 40,
        n_clusters: int = 8,
        lr: float = 0.1,
        temp: float = 0.3,
        # Note: Temperature not directly used in current training/answering logic?
        sample_size: int = 5,
        *,
        seed: Optional[int] = None,
        max_tokens: Optional[int] = None,
        max_grad_norm: float = 1.0,
        init_pool_retries: int = 1,
    ):
        """
        Initializes the CDWCoT instance.

        Args:
            llm: The language model instance to use.
            pool_size: The desired maximum size of the prompt pool (PC).
            n_clusters: The number of clusters to group training questions into.
            lr: Learning rate for updating cluster prompt distributions.
            temp: Temperature parameter (currently unused in core logic).
            sample_size: Number of prompts to sample/select for context/validation.
            seed: Optional random seed for reproducibility.
            max_tokens: Optional maximum number of tokens for LLM generation.
            max_grad_norm: Maximum norm for gradient clipping during training.
            init_pool_retries: Number of retries when generating initial CoTs for the pool.
        """
        self.llm = llm
        self.pool_size = pool_size
        self.n_clusters = n_clusters
        self.lr = lr
        self.temp = temp  # Store temperature, though might not be actively used
        self.sample_size = sample_size
        self.seed = seed
        self.max_tokens = max_tokens
        self.max_grad_norm = max_grad_norm
        self.init_pool_retries = init_pool_retries

        # Internal state variables
        self.cluster_centers: Optional[np.ndarray] = None  # Centroids of question clusters
        self.PC: List[str] = []  # The prompt pool (CoT demonstrations)
        self.p_cluster: List[
            np.ndarray
        ] = []  # Learned probability distributions over PC for each cluster
        self.pool_map: List[
            Tuple[int, str]
        ] = []  # Maps prompts in PC back to original question index/text
        self.train_questions: List[str] = []  # Stored training questions
        self.train_answers: List[str] = []  # Stored training answers
        self.train_labels: List[int] = []  # Cluster labels for each training question

    def _is_valid_distribution(self, p: np.ndarray) -> bool:
        """Checks if a numpy array represents a valid probability distribution."""
        return bool(p.size) and np.all(p >= 0) and np.isclose(p.sum(), 1.0)

    def _select_pool_indices(self, questions: List[str]) -> List[Tuple[int, str]]:
        """
        Clusters questions and selects candidate indices for the prompt pool.

        Selects questions closest to cluster centers, proportionally to cluster size.

        Args:
            questions: List of training questions.

        Returns:
            List of tuples (original_index, question_text) for pool candidates.
        """
        N = len(questions)
        effective_n = min(self.n_clusters, N)
        if effective_n <= 0:
            raise ValueError("Cannot initialize pool with zero clusters")

        logger.info(f"Encoding {N} questions for clustering...")
        embs = np.stack(encode(questions))
        logger.info(f"Clustering embeddings into {effective_n} clusters...")
        labels, centers = cluster_embeddings(embs, effective_n, random_state=self.seed or 0)
        self.cluster_centers = centers
        self.train_labels = labels.tolist()

        # Select candidates: questions closest to centroids, proportional to cluster size
        m: dict[int, str] = {}  # Using dict to avoid duplicates
        for c in range(effective_n):
            idxs = [i for i, lab in enumerate(labels) if lab == c]
            if not idxs:
                logger.debug(f"Cluster {c} has no associated questions.")
                continue

            # Calculate number of samples 'k' for this cluster
            k = (
                min(len(idxs), max(1, int(round(len(idxs) / N * self.pool_size))))
                if self.pool_size > 0
                else 0
            )
            logger.debug(f"Cluster {c} (size {len(idxs)}) sampling k={k} candidates for pool.")
            if k <= 0:
                continue

            # Find k points closest to the cluster center
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

    def init_pool(self, questions: List[str], answers: List[str]) -> None:
        """
        Initializes the prompt pool (PC) synchronously.

        Clusters questions, selects candidates, generates CoT for them using the LLM.

        Args:
            questions: List of training questions.
            answers: List of corresponding training answers.
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

        # Generate CoT for each candidate
        for idx, q in pool_candidates:
            prompt = f"Q: {q}\nA: Let's think step by step."
            cot: Optional[str] = None
            for attempt in range(self.init_pool_retries + 1):
                try:
                    # Generate a unique seed for each attempt if base seed is provided
                    gen_seed = (
                        (self.seed + idx * (self.init_pool_retries + 1) + attempt)
                        if self.seed is not None
                        else None
                    )
                    cot = self.llm.generate(prompt, max_tokens=self.max_tokens, seed=gen_seed)
                    cots[idx] = f"Q: {q}\nA: {cot}"  # Store the full Q/A CoT pair
                    successful_indices.append(idx)
                    logger.debug(f"Successfully generated CoT for pool candidate index {idx}.")
                    break  # Success, move to next candidate
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed for pool index {idx}: {e}")
                    if attempt < self.init_pool_retries:
                        time.sleep(0.5 * 2**attempt)  # Exponential backoff
                    else:
                        logger.error(
                            "Failed to generate CoT for pool index %d ('%s') after %d retries: %s",
                            idx,
                            q[:50] + "...",
                            self.init_pool_retries + 1,
                            e,
                        )
                        failed_indices.append(idx)

        # Populate the prompt pool (PC) and the map
        self.PC = [cots[idx] for idx, _ in pool_candidates if idx in successful_indices]
        self.pool_map = [(idx, q) for idx, q in pool_candidates if idx in successful_indices]
        M = len(self.PC)

        if M == 0:
            raise RuntimeError("Prompt pool is empty after init_pool - all CoT generations failed.")
        elif failed_indices:
            logger.warning(f"Failed to generate CoT for {len(failed_indices)} pool candidates.")

        # Initialize cluster probabilities to uniform
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
    ) -> None:
        """
        Initializes the prompt pool (PC) asynchronously.

        Clusters questions, selects candidates, generates CoT for them using the LLM concurrently.

        Args:
            questions: List of training questions.
            answers: List of corresponding training answers.
            semaphore: Optional semaphore to limit concurrent LLM calls.
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

        # Async function to generate CoT for one candidate
        async def gen(idx: int, q: str) -> Tuple[int, Optional[str]]:
            prompt = f"Q: {q}\nA: Let's think step by step."
            for attempt in range(self.init_pool_retries + 1):
                try:
                    # Generate a unique seed for each attempt if base seed is provided
                    gen_seed = (
                        (self.seed + idx * (self.init_pool_retries + 1) + attempt)
                        if self.seed is not None
                        else None
                    )
                    # Use semaphore if provided
                    if semaphore:
                        async with semaphore:
                            cot = await self.llm.generate_async(
                                prompt, max_tokens=self.max_tokens, seed=gen_seed
                            )
                    else:
                        cot = await self.llm.generate_async(
                            prompt, max_tokens=self.max_tokens, seed=gen_seed
                        )
                    logger.debug(
                        f"Successfully generated async CoT for pool candidate index {idx}."
                    )
                    return idx, cot  # Success
                except Exception as e:
                    logger.warning(f"Async attempt {attempt + 1} failed for pool index {idx}: {e}")
                    if attempt < self.init_pool_retries:
                        await asyncio.sleep(0.5 * 2**attempt)  # Exponential backoff
                    else:
                        logger.error(
                            "Failed async CoT gen for pool index %d ('%s') after %d retries: %s",
                            idx,
                            q[:50] + "...",
                            self.init_pool_retries + 1,
                            e,
                        )
                        return idx, None  # Failure after retries

        # Run generation tasks concurrently
        tasks = [gen(idx, q) for idx, q in pool_candidates]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        cots: dict[int, str] = {}
        successful_indices: List[int] = []
        failed_indices: List[int] = []
        for i, res in enumerate(results):
            original_index = pool_candidates[i][0]
            original_question = pool_candidates[i][1]
            if isinstance(res, Exception):
                logger.error(f"Async generation task failed for index {original_index}: {res}")
                failed_indices.append(original_index)
            elif isinstance(res, tuple) and len(res) == 2:
                idx, cot_result = res
                if cot_result is not None:
                    # Use the original question text stored earlier
                    cots[idx] = f"Q: {self.train_questions[idx]}\nA: {cot_result}"
                    successful_indices.append(idx)
                else:
                    failed_indices.append(idx)  # Generation failed after retries
            else:
                logger.error(
                    f"Unexpected result type from async generation task for index {original_index}: {type(res)}"
                )
                failed_indices.append(original_index)

        # Populate the prompt pool (PC) and the map
        # Ensure correct mapping using successful_indices
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

        # Initialize cluster probabilities to uniform
        num_cl = self.cluster_centers.shape[0] if self.cluster_centers is not None else 0
        self.p_cluster = [np.ones(M) / M for _ in range(num_cl)]
        logger.info(
            f"Initialized prompt pool asynchronously with {M} prompts and {num_cl} uniform cluster distributions."
        )

    def train(self, val_split: float = 0.2, epochs: int = 100, patience: int = 5) -> None:
        """
        Trains the prompt distributions for each cluster synchronously.

        Uses a policy gradient-like approach based on exact match accuracy.

        Args:
            val_split: Fraction of data per cluster to use for validation.
            epochs: Maximum number of training epochs per cluster.
            patience: Number of epochs to wait for improvement before early stopping.
        """
        if not self.PC or self.cluster_centers is None or not self.p_cluster:
            raise RuntimeError("Call init_pool first before training.")

        logger.info("Starting synchronous training...")
        rnd = np.random.RandomState(self.seed) if self.seed is not None else np.random.RandomState()
        M = len(self.PC)
        nc = len(self.p_cluster)  # Should match self.cluster_centers.shape[0]

        # Group training indices by cluster
        cluster_idxs = {
            c: [i for i, lab in enumerate(self.train_labels) if lab == c] for c in range(nc)
        }

        for c, idxs in cluster_idxs.items():
            if not idxs:  # Skip clusters with no assigned training data
                logger.debug(f"Skipping training for empty cluster {c}.")
                # Ensure p_cluster exists and remains uniform if empty
                if c < len(self.p_cluster) and not self._is_valid_distribution(self.p_cluster[c]):
                    self.p_cluster[c] = np.ones(M) / M
                continue

            # Split data for this cluster
            rnd.shuffle(idxs)
            split_idx = max(
                1, int(len(idxs) * (1 - val_split))
            )  # Ensure at least one train/val sample if possible
            train_idx, val_idx = idxs[:split_idx], idxs[split_idx:]

            # Handle cases where split results in empty validation set
            if not val_idx:
                logger.warning(
                    "Validation set empty for cluster %d (size %d, split %f). Using training set for validation.",
                    c,
                    len(idxs),
                    val_split,
                )
                val_idx = train_idx  # Use training data for validation as fallback
            if not train_idx:
                logger.warning(
                    f"Training set empty for cluster {c}. Skipping training for this cluster."
                )
                continue

            logger.info(
                f"Training cluster {c}: {len(train_idx)} train samples, {len(val_idx)} validation samples."
            )

            # Get initial probability distribution for the cluster
            p = self.p_cluster[c].copy()
            if not self._is_valid_distribution(p):
                logger.warning(
                    f"Initial distribution for cluster {c} invalid, resetting to uniform."
                )
                p = np.ones(M) / M

            best_p = p.copy()
            best_acc = -1.0
            no_imp = 0

            # Training loop for the cluster
            for epoch in range(epochs):
                # Sample a batch from training data
                batch = rnd.choice(
                    train_idx, size=min(len(train_idx), self.sample_size), replace=False
                )

                losses: List[float] = []
                grads = np.zeros_like(p)
                batch_results: List[Tuple[int, float]] = []  # Store (prompt_index, loss)

                # Process batch items
                for j, orig_idx in enumerate(batch):
                    # Sample a prompt from the pool based on current distribution p
                    m = rnd.choice(M, p=p)
                    q = self.train_questions[orig_idx]
                    prev = self.PC[m]  # The sampled CoT demonstration
                    payload = f"{prev}\n\nQ: {q}\nA: Let's think step by step."
                    loss = 1.0  # Assume failure unless proven otherwise

                    try:
                        gen_seed = (self.seed or 0) + epoch * len(batch) + j
                        out = self.llm.generate(
                            payload,
                            max_tokens=self.max_tokens,
                            seed=gen_seed,
                        )
                        # Calculate loss (0 for correct, 1 for incorrect)
                        if exact_match(out, self.train_answers[orig_idx]):
                            loss = 0.0
                    except Exception as e:
                        logger.error(
                            f"Sync train generation failed for q_idx {orig_idx}, p_idx {m}: {e}"
                        )
                        # Keep loss = 1.0

                    batch_results.append((m, loss))
                    losses.append(loss)

                if not losses:  # Should not happen if batch has items, but safeguard
                    continue

                # Calculate policy gradient estimate
                mean_loss = np.mean(losses)
                for m_idx, loss in batch_results:
                    advantage = (
                        loss - mean_loss
                    )  # Lower loss is better -> negative advantage is good
                    # Gradient ascends towards lower loss (prompts with negative advantage)
                    # Divide by probability to counteract sampling bias
                    grads[m_idx] += -advantage / max(p[m_idx], 1e-9)

                # Clip gradients
                norm = np.linalg.norm(grads)
                if norm > self.max_grad_norm:
                    grads *= self.max_grad_norm / norm

                # Update probabilities (gradient ascent on reward = -loss)
                # Normalize gradient by batch size
                p = p - self.lr * (grads / len(losses))  # Subtract gradient * lr
                p = np.clip(p, 1e-9, None)  # Ensure non-negative probabilities

                # Re-normalize to ensure it's a valid distribution
                p_sum = p.sum()
                p = p / p_sum if p_sum > 1e-9 else np.ones_like(p) / p.size

                # Validation step
                val_preds = []
                for val_orig in val_idx:
                    # Select top prompts based on the *updated* distribution p for context
                    top_indices = np.argsort(-p)[: min(self.sample_size, M)]
                    ctx = "\n\n".join(self.PC[i] for i in top_indices)
                    vp = f"{ctx}\n\nQ: {self.train_questions[val_orig]}\nA: Let's think step by step."
                    val_out = ""
                    try:
                        val_seed = (
                            self.seed or 0
                        ) + val_orig  # Consistent seed per validation question
                        val_out = self.llm.generate(vp, max_tokens=self.max_tokens, seed=val_seed)
                    except Exception as e:
                        logger.error(f"Sync validation generation failed for q_idx {val_orig}: {e}")
                        val_out = "[ERROR]"  # Mark as error for accuracy calc
                    val_preds.append(val_out)

                # Calculate validation accuracy
                val_golds = [self.train_answers[i] for i in val_idx]
                acc = accuracy(val_preds, val_golds)
                logger.debug(
                    f"Cluster {c} Epoch {epoch + 1}: Train Loss={mean_loss:.3f}, Val Acc={acc:.3f}"
                )

                # Early stopping check
                if acc > best_acc:
                    best_acc, best_p, no_imp = acc, p.copy(), 0
                else:
                    no_imp += 1
                    if no_imp >= patience:
                        logger.info(
                            f"Early stopping for cluster {c} at epoch {epoch + 1} (Val Acc: {best_acc:.3f})"
                        )
                        break

            # Update the cluster's distribution with the best one found
            self.p_cluster[c] = best_p
            logger.info(f"Finished training cluster {c}. Best Val Acc: {best_acc:.3f}")

    # ************************************************
    # ****** FIXED train_async Implementation ******
    # ************************************************
    async def train_async(
        self,
        val_split: float = 0.2,
        epochs: int = 100,
        patience: int = 5,
        semaphore: Optional[asyncio.Semaphore] = None,
    ) -> None:
        """
        Trains the prompt distributions for each cluster asynchronously.

        Uses a policy gradient-like approach based on exact match accuracy,
        performing LLM calls concurrently.

        Args:
            val_split: Fraction of data per cluster to use for validation.
            epochs: Maximum number of training epochs per cluster.
            patience: Number of epochs to wait for improvement before early stopping.
            semaphore: Optional semaphore to limit concurrent LLM calls.
        """
        if not self.PC or self.cluster_centers is None or not self.p_cluster:
            raise RuntimeError("Call init_pool_async first before training.")

        logger.info("Starting asynchronous training...")
        rnd = np.random.RandomState(self.seed) if self.seed is not None else np.random.RandomState()
        M = len(self.PC)
        nc = len(self.p_cluster)

        # Group training indices by cluster
        cluster_idxs = {
            c: [i for i, lab in enumerate(self.train_labels) if lab == c] for c in range(nc)
        }

        training_coroutines = []

        # Create a training coroutine for each cluster
        for c, idxs in cluster_idxs.items():
            if not idxs:
                logger.debug(f"Skipping async training for empty cluster {c}.")
                if c < len(self.p_cluster) and not self._is_valid_distribution(self.p_cluster[c]):
                    self.p_cluster[c] = np.ones(M) / M
                continue

            # Split data for this cluster
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

            # Define the async function to train one cluster
            async def train_cluster(
                cluster_index: int,
                initial_p: np.ndarray,
                train_indices: List[int],
                val_indices: List[int],
            ):
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

                    # Async function to process one item in the batch
                    async def process_batch_item(j: int, orig_idx: int) -> Tuple[int, float]:
                        # Sample prompt based on current distribution p
                        m = rnd.choice(M, p=p)
                        q = self.train_questions[orig_idx]
                        prev = self.PC[m]
                        payload = f"{prev}\n\nQ: {q}\nA: Let's think step by step."
                        loss = 1.0  # Default loss is 1 (incorrect)
                        try:
                            gen_seed = (self.seed or 0) + epoch * len(batch_indices) + j
                            # Perform async LLM call with semaphore
                            if semaphore:
                                async with semaphore:
                                    out = await self.llm.generate_async(
                                        payload, max_tokens=self.max_tokens, seed=gen_seed
                                    )
                            else:
                                out = await self.llm.generate_async(
                                    payload, max_tokens=self.max_tokens, seed=gen_seed
                                )

                            # Check result and set loss
                            if exact_match(out, self.train_answers[orig_idx]):
                                loss = 0.0
                        except Exception as e:
                            logger.error(
                                f"Async train generation failed q_idx {orig_idx}, p_idx {m}: {e}"
                            )
                        return m, loss  # Return (prompt_index, loss)

                    # Run batch processing concurrently
                    batch_results_tuples: List[Tuple[int, float]] = await asyncio.gather(
                        *(
                            process_batch_item(j, orig_idx)
                            for j, orig_idx in enumerate(batch_indices)
                        )
                    )

                    losses = [loss for _, loss in batch_results_tuples]
                    if not losses:
                        continue

                    # Calculate gradients (same logic as sync version)
                    mean_loss = np.mean(losses)
                    grads = np.zeros_like(p)
                    for m_idx, loss in batch_results_tuples:
                        advantage = loss - mean_loss
                        grads[m_idx] += -advantage / max(p[m_idx], 1e-9)

                    norm = np.linalg.norm(grads)
                    if norm > self.max_grad_norm:
                        grads *= self.max_grad_norm / norm

                    # Update probabilities (same logic as sync version)
                    p = p - self.lr * (grads / len(losses))
                    p = np.clip(p, 1e-9, None)
                    p_sum = p.sum()
                    p = p / p_sum if p_sum > 1e-9 else np.ones_like(p) / p.size

                    # --- Asynchronous Validation Step ---
                    async def validate_item(val_orig_idx: int) -> str:
                        # Select top prompts based on updated p
                        top_indices = np.argsort(-p)[: min(self.sample_size, M)]
                        ctx = "\n\n".join(self.PC[i] for i in top_indices)
                        vp = f"{ctx}\n\nQ: {self.train_questions[val_orig_idx]}\nA: Let's think step by step."
                        val_out = "[ERROR]"
                        try:
                            val_seed = (self.seed or 0) + val_orig_idx
                            # Perform async LLM call for validation
                            if semaphore:
                                async with semaphore:
                                    val_out = await self.llm.generate_async(
                                        vp, max_tokens=self.max_tokens, seed=val_seed
                                    )
                            else:
                                val_out = await self.llm.generate_async(
                                    vp, max_tokens=self.max_tokens, seed=val_seed
                                )
                        except Exception as e:
                            logger.error(
                                f"Async validation generation failed q_idx {val_orig_idx}: {e}"
                            )
                        return val_out

                    # Run validation concurrently
                    val_preds = await asyncio.gather(
                        *(validate_item(val_idx) for val_idx in val_indices)
                    )

                    # Calculate accuracy
                    val_golds = [self.train_answers[i] for i in val_indices]
                    acc = accuracy(val_preds, val_golds)
                    logger.debug(
                        f"Async Cluster {cluster_index} Epoch {epoch + 1}: Train Loss={mean_loss:.3f}, Val Acc={acc:.3f}"
                    )

                    # Early stopping check
                    if acc > best_acc:
                        best_acc, best_p, no_imp = acc, p.copy(), 0
                    else:
                        no_imp += 1
                        if no_imp >= patience:
                            logger.info(
                                f"Async early stopping for cluster {cluster_index} at epoch {epoch + 1} (Val Acc: {best_acc:.3f})"
                            )
                            break

                # Store the best distribution found for this cluster
                self.p_cluster[cluster_index] = best_p
                logger.info(
                    f"Finished async training cluster {cluster_index}. Best Val Acc: {best_acc:.3f}"
                )

            # Add the cluster training coroutine to the list
            training_coroutines.append(train_cluster(c, self.p_cluster[c], train_idx, val_idx))

        # Run training for all clusters concurrently
        await asyncio.gather(*training_coroutines)
        logger.info("Asynchronous CDW-CoT training complete for all clusters.")

    def _calculate_combined_distribution(self, question: str) -> np.ndarray:
        M = len(self.PC)
        if M == 0:
            raise RuntimeError("Prompt pool (PC) is empty. Cannot calculate distribution.")

        if self.cluster_centers is None or not self.p_cluster:
            logger.warning(
                "Cluster centers or probabilities not initialized. Falling back to uniform distribution."
            )
            return np.ones(M) / M

        try:
            logger.debug(f"Encoding question for distribution calculation: '{question[:50]}...'")
            q_emb_list = encode([question])

            # --- FIXED Truthiness Check ---
            # Explicitly check length instead of relying on 'not list'
            if len(q_emb_list) == 0 or q_emb_list[0] is None:
                raise ValueError("Encoding failed or returned None for the input question.")
            # --- End Fixed Check ---

            q_emb = np.stack(q_emb_list)[0]

            if q_emb is not None and self.cluster_centers.size > 0:
                if q_emb.shape != self.cluster_centers.shape[1:]:
                    q_emb = q_emb.reshape(1, -1)
                    if q_emb.shape[1] != self.cluster_centers.shape[1]:
                        raise ValueError(
                            f"Question embedding dimension {q_emb.shape} doesn't match cluster center dimension {self.cluster_centers.shape}"
                        )

                dists = np.linalg.norm(self.cluster_centers - q_emb, axis=1)
                closest_cluster_idx = np.argmin(dists)
                logger.debug(f"Question mapped to closest cluster index: {closest_cluster_idx}")

                if 0 <= closest_cluster_idx < len(self.p_cluster):
                    learned_p = self.p_cluster[closest_cluster_idx]
                    if self._is_valid_distribution(learned_p):
                        logger.debug(
                            f"Using learned distribution for cluster {closest_cluster_idx}"
                        )
                        return learned_p
                    else:
                        logger.warning(
                            f"Invalid learned distribution found for cluster {closest_cluster_idx}. Falling back to uniform."
                        )
                else:
                    logger.warning(
                        f"Closest cluster index {closest_cluster_idx} out of bounds for p_cluster (size {len(self.p_cluster)}). Falling back to uniform."
                    )
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

    def answer(self, test_q: str) -> str:
        """
        Answers a test question synchronously using the learned distributions.

        Args:
            test_q: The question to answer.

        Returns:
            The generated answer string.
        """
        # Get the appropriate distribution (FIXED: now uses learned probs)
        dist = self._calculate_combined_distribution(test_q)
        M = len(self.PC)
        if M == 0:
            raise RuntimeError("Prompt pool is empty, cannot answer.")

        # Select top prompts based on the calculated distribution
        top_indices = np.argsort(-dist)[: min(self.sample_size, M)]
        logger.debug(
            f"Selected top prompt indices for sync answer: {top_indices} based on distribution."
        )
        ctxt = "\n\n".join(self.PC[i] for i in top_indices)
        payload = f"{ctxt}\n\nQ: {test_q}\nA: Let's think step by step."

        # Generate answer with a potentially consistent seed
        gen_seed = (self.seed + len(self.train_questions)) if self.seed is not None else None
        logger.info(f"Generating sync answer for: '{test_q[:50]}...'")
        return self.llm.generate(payload, max_tokens=self.max_tokens, seed=gen_seed)

    async def answer_async(self, test_q: str, semaphore: Optional[asyncio.Semaphore] = None) -> str:
        """
        Answers a test question asynchronously using the learned distributions.

        Args:
            test_q: The question to answer.
            semaphore: Optional semaphore to limit concurrent LLM calls.

        Returns:
            The generated answer string.
        """
        # Get the appropriate distribution (FIXED: now uses learned probs)
        dist = self._calculate_combined_distribution(test_q)
        M = len(self.PC)
        if M == 0:
            raise RuntimeError("Prompt pool is empty, cannot answer.")

        # Select top prompts based on the calculated distribution
        top_indices = np.argsort(-dist)[: min(self.sample_size, M)]
        logger.debug(
            f"Selected top prompt indices for async answer: {top_indices} based on distribution."
        )
        ctxt = "\n\n".join(self.PC[i] for i in top_indices)
        payload = f"{ctxt}\n\nQ: {test_q}\nA: Let's think step by step."

        # Generate answer with a potentially consistent seed
        gen_seed = (self.seed + len(self.train_questions)) if self.seed is not None else None
        logger.info(f"Generating async answer for: '{test_q[:50]}...'")

        # Use semaphore if provided
        if semaphore:
            async with semaphore:
                return await self.llm.generate_async(
                    payload, max_tokens=self.max_tokens, seed=gen_seed
                )
        else:
            return await self.llm.generate_async(payload, max_tokens=self.max_tokens, seed=gen_seed)

    # Allow calling the instance like a function (defaults to synchronous answer)
    __call__ = answer
