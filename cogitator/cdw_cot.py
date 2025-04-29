import asyncio
import logging
import time
from typing import List, Optional, Tuple

import numpy as np

from .model import BaseLLM
from .utils import accuracy, cluster_embeddings, encode, exact_match

logger = logging.getLogger(__name__)


class CDWCoT:
    def __init__(
        self,
        llm: BaseLLM,
        pool_size: int = 40,
        n_clusters: int = 8,
        lr: float = 0.1,
        temp: float = 0.3,
        sample_size: int = 5,
        seed: Optional[int] = None,
        max_grad_norm: float = 1.0,
        init_pool_retries: int = 1,
    ):
        self.llm = llm
        self.pool_size = pool_size
        self.n_clusters = n_clusters
        self.lr = lr
        self.temp = temp
        self.sample_size = sample_size
        self.seed = seed
        self.max_grad_norm = max_grad_norm
        self.init_pool_retries = init_pool_retries
        self.cluster_centers: Optional[np.ndarray] = None
        self.PC: List[str] = []
        self.p_cluster: List[np.ndarray] = []
        self.pool_map: List[Tuple[int, str]] = []
        self.train_questions: List[str] = []
        self.train_answers: List[str] = []
        self.train_labels: List[int] = []

    def _is_valid_distribution(self, p: np.ndarray) -> bool:
        return bool(p.size) and np.all(p >= 0) and np.isclose(p.sum(), 1.0)

    def _select_pool_indices(self, questions: List[str]) -> List[Tuple[int, str]]:
        N = len(questions)
        effective_n = min(self.n_clusters, N)
        if effective_n <= 0:
            raise ValueError("Cannot initialize pool with zero clusters")
        embs = np.stack(encode(questions))
        labels, centers = cluster_embeddings(embs, effective_n)
        self.cluster_centers = centers
        self.train_labels = labels.tolist()
        m: dict[int, str] = {}
        for c in range(effective_n):
            idxs = [i for i, lab in enumerate(labels) if lab == c]
            if not idxs:
                continue
            k = (
                min(len(idxs), max(1, int(round(len(idxs) / N * self.pool_size))))
                if self.pool_size > 0
                else 0
            )
            if k <= 0:
                continue
            d = np.linalg.norm(embs[idxs] - centers[c], axis=1)
            for i in np.argsort(d)[:k]:
                m.setdefault(idxs[i], questions[idxs[i]])
        return sorted(m.items())

    def init_pool(self, questions: List[str], answers: List[str]) -> None:
        if len(questions) != len(answers):
            raise ValueError("questions/answers length mismatch")
        self.train_questions = questions
        self.train_answers = answers
        pool_candidates = self._select_pool_indices(questions)
        if not pool_candidates:
            raise RuntimeError("Prompt pool selection resulted in zero candidates")

        cots: dict[int, str] = {}
        successful_indices: List[int] = []
        failed_indices: List[int] = []

        for idx, q in pool_candidates:
            prompt = f"Q: {q}\nA: Let's think step by step."
            cot = None
            for attempt in range(self.init_pool_retries + 1):
                try:
                    seed = (
                        self.seed + idx * (self.init_pool_retries + 1) + attempt
                        if self.seed is not None
                        else None
                    )
                    cot = self.llm.generate(prompt, seed=seed)
                    cots[idx] = f"Q: {q}\nA: {cot}"
                    successful_indices.append(idx)
                    break
                except Exception as e:
                    if attempt < self.init_pool_retries:
                        time.sleep(0.5 * 2**attempt)
                    else:
                        logger.error(
                            "Failed to generate CoT for pool index %d ('%s') after %d retries: %s",
                            idx,
                            q,
                            self.init_pool_retries + 1,
                            e,
                        )
                        failed_indices.append(idx)

        self.PC = [cots[idx] for idx, _ in pool_candidates if idx in successful_indices]
        self.pool_map = [(idx, q) for idx, q in pool_candidates if idx in successful_indices]

        M = len(self.PC)
        if M == 0:
            raise RuntimeError("Prompt pool empty after init_pool - all generations failed.")
        elif failed_indices:
            logger.warning("Failed to generate CoT for %d pool candidates.", len(failed_indices))

        num_cl = self.cluster_centers.shape[0]
        self.p_cluster = [np.ones(M) / M for _ in range(num_cl)]

    async def init_pool_async(
        self,
        questions: List[str],
        answers: List[str],
        semaphore: Optional[asyncio.Semaphore] = None,
    ) -> None:
        if len(questions) != len(answers):
            raise ValueError("questions/answers length mismatch")
        self.train_questions = questions
        self.train_answers = answers
        pool_candidates = self._select_pool_indices(questions)
        if not pool_candidates:
            raise RuntimeError("Prompt pool selection resulted in zero candidates")

        async def gen(idx: int, q: str):
            prompt = f"Q: {q}\nA: Let's think step by step."
            for attempt in range(self.init_pool_retries + 1):
                try:
                    seed = (
                        self.seed + idx * (self.init_pool_retries + 1) + attempt
                        if self.seed is not None
                        else None
                    )
                    if semaphore:
                        async with semaphore:
                            cot = await self.llm.generate_async(prompt, seed=seed)
                    else:
                        cot = await self.llm.generate_async(prompt, seed=seed)
                    return idx, f"Q: {q}\nA: {cot}"
                except Exception as e:
                    if attempt < self.init_pool_retries:
                        await asyncio.sleep(0.5 * 2**attempt)
                    else:
                        logger.error(
                            "Failed async CoT gen for pool index %d ('%s') after %d retries: %s",
                            idx,
                            q,
                            self.init_pool_retries + 1,
                            e,
                        )
                        return idx, None  # Indicate failure

        tasks = [gen(idx, q) for idx, q in pool_candidates]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        cots: dict[int, str] = {}
        successful_indices: List[int] = []
        failed_indices: List[int] = []

        for res in results:
            if isinstance(res, Exception):
                logger.error("Async CoT generation task failed: %s", res)
                # Can't easily get the original index here without more complex tracking
                continue
            if res is None:  # Should not happen if gen always returns tuple
                continue

            idx, cot_result = res
            if cot_result is not None:
                cots[idx] = cot_result
                successful_indices.append(idx)
            else:
                failed_indices.append(idx)

        self.PC = [cots[idx] for idx, _ in pool_candidates if idx in successful_indices]
        self.pool_map = [(idx, q) for idx, q in pool_candidates if idx in successful_indices]

        M = len(self.PC)
        if M == 0:
            raise RuntimeError("Prompt pool empty after async init_pool - all generations failed.")
        elif failed_indices:
            logger.warning(
                "Failed to generate async CoT for %d pool candidates.", len(failed_indices)
            )

        num_cl = self.cluster_centers.shape[0]
        self.p_cluster = [np.ones(M) / M for _ in range(num_cl)]

    def train(self, val_split: float = 0.2, epochs: int = 100, patience: int = 5) -> None:
        if not self.PC or self.cluster_centers is None:
            raise RuntimeError("Call init_pool first")
        rnd = np.random.RandomState(self.seed) if self.seed is not None else np.random.RandomState()
        M = len(self.PC)
        nc = len(self.p_cluster)
        cluster_idxs = {
            c: [i for i, lab in enumerate(self.train_labels) if lab == c] for c in range(nc)
        }
        for c, idxs in cluster_idxs.items():
            if not idxs:
                self.p_cluster[c] = np.ones(M) / M
                continue
            rnd.shuffle(idxs)
            split_idx = max(1, int(len(idxs) * (1 - val_split)))
            train_idx, val_idx = idxs[:split_idx], idxs[split_idx:]
            if not val_idx:
                logger.warning(
                    "Validation set empty for cluster %d, using training set for validation.", c
                )
                val_idx = train_idx

            p = self.p_cluster[c].copy()
            if not self._is_valid_distribution(p):
                p = np.ones(M) / M
            best_p, best_acc, no_imp = p.copy(), -1.0, 0
            for epoch in range(epochs):
                batch = rnd.choice(
                    train_idx, size=min(len(train_idx), self.sample_size), replace=False
                )
                losses, grads = [], np.zeros_like(p)
                batch_results = []
                for j, orig_idx in enumerate(batch):
                    m = rnd.choice(M, p=p)
                    q = self.train_questions[orig_idx]
                    prev = self.PC[m]
                    payload = f"{prev}\n\nQ: {q}\nA: Let's think step by step."
                    try:
                        resp = self.llm.generate(
                            payload, seed=(self.seed or 0) + epoch * len(batch) + j
                        )
                        loss = 0.0 if exact_match(resp, self.train_answers[orig_idx]) else 1.0
                    except Exception as e:
                        logger.warning(
                            "Error during training generation for q_idx %d: %s", orig_idx, e
                        )
                        loss = 1.0
                    batch_results.append((m, loss))
                    losses.append(loss)

                if not losses:
                    logger.warning(
                        "No losses calculated in epoch %d for cluster %d, skipping update.",
                        epoch,
                        c,
                    )
                    continue

                mean_loss = np.mean(losses)
                for m, loss in batch_results:
                    adv = loss - mean_loss
                    grads[m] += -adv / max(p[m], 1e-9)

                norm = np.linalg.norm(grads)
                if norm > self.max_grad_norm:
                    grads *= self.max_grad_norm / norm
                p = np.clip(p - self.lr * (grads / len(losses)), 1e-9, None)
                p_sum = p.sum()
                p = p / p_sum if p_sum > 1e-9 else np.ones_like(p) / p.size

                val_preds = []
                for val_orig_idx in val_idx:
                    qv = self.train_questions[val_orig_idx]
                    top = np.argsort(-p)[: min(self.sample_size, M)]
                    ctx = "\n\n".join(self.PC[i] for i in top)
                    vp = f"{ctx}\n\nQ: {qv}\nA: Let's think step by step."
                    try:
                        out = self.llm.generate(vp, seed=(self.seed or 0) + val_orig_idx)
                    except Exception as e:
                        logger.warning(
                            "Error during validation generation for q_idx %d: %s", val_orig_idx, e
                        )
                        out = ""
                    val_preds.append(out)

                acc = accuracy(val_preds, [self.train_answers[i] for i in val_idx])
                if acc > best_acc:
                    best_acc, best_p, no_imp = acc, p.copy(), 0
                else:
                    no_imp += 1
                    if no_imp >= patience:
                        logger.info("Early stopping training for cluster %d at epoch %d", c, epoch)
                        break
            self.p_cluster[c] = best_p

    async def train_async(
        self,
        val_split: float = 0.2,
        epochs: int = 100,
        patience: int = 5,
        semaphore: Optional[asyncio.Semaphore] = None,
    ) -> None:
        if not self.PC or self.cluster_centers is None:
            raise RuntimeError("Call init_pool first")
        rnd = np.random.RandomState(self.seed) if self.seed is not None else np.random.RandomState()
        M = len(self.PC)
        nc = len(self.p_cluster)
        cluster_idxs = {
            c: [i for i, lab in enumerate(self.train_labels) if lab == c] for c in range(nc)
        }

        async def train_cluster(c: int, idxs: List[int]):
            if not idxs:
                return np.ones(M) / M

            rnd.shuffle(idxs)
            split_idx = max(1, int(len(idxs) * (1 - val_split)))
            train_idx, val_idx = idxs[:split_idx], idxs[split_idx:]
            if not val_idx:
                logger.warning(
                    "Validation set empty for cluster %d, using training set for validation.", c
                )
                val_idx = train_idx

            p = self.p_cluster[c].copy()
            if not self._is_valid_distribution(p):
                p = np.ones(M) / M
            best_p, best_acc, no_imp = p.copy(), -1.0, 0

            for epoch in range(epochs):
                batch = rnd.choice(
                    train_idx, size=min(len(train_idx), self.sample_size), replace=False
                )
                tasks = []

                async def run_train_gen(m_idx, orig_idx, payload_str, seed_val):
                    try:
                        if semaphore:
                            async with semaphore:
                                out = await self.llm.generate_async(payload_str, seed=seed_val)
                        else:
                            out = await self.llm.generate_async(payload_str, seed=seed_val)
                        loss_val = 0.0 if exact_match(out, self.train_answers[orig_idx]) else 1.0
                        return m_idx, loss_val
                    except Exception as e:
                        logger.warning(
                            "Error during async training generation for q_idx %d: %s", orig_idx, e
                        )
                        return m_idx, 1.0

                for j, orig in enumerate(batch):
                    m = rnd.choice(M, p=p)
                    q = self.train_questions[orig]
                    prev = self.PC[m]
                    payload = f"{prev}\n\nQ: {q}\nA: Let's think step by step."
                    seed = (self.seed or 0) + epoch * len(batch) + j
                    tasks.append(run_train_gen(m, orig, payload, seed))

                results = await asyncio.gather(*tasks)
                losses, grads = [], np.zeros_like(p)
                for m_res, loss_res in results:
                    losses.append(loss_res)

                if not losses:
                    logger.warning(
                        "No losses calculated in async epoch %d for cluster %d, skipping update.",
                        epoch,
                        c,
                    )
                    continue

                mean_loss = np.mean(losses)
                for m_res, loss_res in results:
                    adv = loss_res - mean_loss
                    grads[m_res] += -adv / max(p[m_res], 1e-9)

                norm = np.linalg.norm(grads)
                if norm > self.max_grad_norm:
                    grads *= self.max_grad_norm / norm
                p = np.clip(p - self.lr * (grads / len(losses)), 1e-9, None)
                p_sum = p.sum()
                p = p / p_sum if p_sum > 1e-9 else np.ones_like(p) / p.size

                val_tasks = []

                async def run_val_gen(payload_str, seed_val):
                    try:
                        if semaphore:
                            async with semaphore:
                                return await self.llm.generate_async(payload_str, seed=seed_val)
                        else:
                            return await self.llm.generate_async(payload_str, seed=seed_val)
                    except Exception as e:
                        logger.warning("Error during async validation generation: %s", e)
                        return ""

                for val_orig_idx in val_idx:
                    qv = self.train_questions[val_orig_idx]
                    top = np.argsort(-p)[: min(self.sample_size, M)]
                    ctx = "\n\n".join(self.PC[i] for i in top)
                    vp = f"{ctx}\n\nQ: {qv}\nA: Let's think step by step."
                    seed = (self.seed or 0) + val_orig_idx
                    val_tasks.append(run_val_gen(vp, seed))

                val_preds = await asyncio.gather(*val_tasks)
                acc = accuracy(val_preds, [self.train_answers[i] for i in val_idx])
                if acc > best_acc:
                    best_acc, best_p, no_imp = acc, p.copy(), 0
                else:
                    no_imp += 1
                    if no_imp >= patience:
                        logger.info(
                            "Early stopping async training for cluster %d at epoch %d", c, epoch
                        )
                        break
            return best_p

        train_tasks = [train_cluster(c, idxs) for c, idxs in cluster_idxs.items()]
        updated_p_clusters = await asyncio.gather(*train_tasks)

        for i, p_new in enumerate(updated_p_clusters):
            cluster_id = list(cluster_idxs.keys())[i]  # Assumes order is preserved
            self.p_cluster[cluster_id] = p_new

    def _calculate_combined_distribution(self, test_q: str) -> np.ndarray:
        if not self.PC or self.cluster_centers is None or not self.p_cluster:
            raise RuntimeError("Model not initialized or trained")
        num_cl = len(self.p_cluster)
        M = len(self.PC)
        weights = np.ones(num_cl) / num_cl if self.temp <= 0 else np.zeros(num_cl)
        if self.temp > 0:
            try:
                e = encode([test_q])[0]
                d = np.linalg.norm(self.cluster_centers - e, axis=1)
                with np.errstate(over="ignore"):  # Ignore overflow in exp
                    w = np.exp(-d / self.temp)
                if not np.isfinite(w.sum()) or w.sum() <= 1e-9:
                    logger.warning(
                        "Cluster weights calculation resulted in non-finite sum or zero, falling back to uniform."
                    )
                    w = np.ones(num_cl)
                weights = w / w.sum()
            except Exception as ex:
                logger.error("Error calculating cluster weights: %s. Falling back to uniform.", ex)
                weights = np.ones(num_cl) / num_cl

        combined = np.zeros(M)
        for i, pc in enumerate(self.p_cluster):
            if weights[i] > 1e-9 and self._is_valid_distribution(pc):
                combined += weights[i] * pc

        combined_sum = combined.sum()
        if combined_sum <= 1e-9:
            logger.warning("Combined distribution sum is near zero, falling back to uniform.")
            combined = np.ones(M) / M
        else:
            combined /= combined_sum
        return combined

    def answer(self, test_q: str) -> str:
        dist = self._calculate_combined_distribution(test_q)
        M = len(self.PC)
        if M == 0:
            raise RuntimeError("Prompt pool is empty")
        top = np.argsort(-dist)[: min(self.sample_size, M)]
        ctxt = "\n\n".join(self.PC[i] for i in top)
        payload = f"{ctxt}\n\nQ: {test_q}\nA: Let's think step by step."
        seed = self.seed + len(self.train_questions) if self.seed is not None else None
        return self.llm.generate(payload, seed=seed)

    async def answer_async(self, test_q: str, semaphore: Optional[asyncio.Semaphore] = None) -> str:
        dist = self._calculate_combined_distribution(test_q)
        M = len(self.PC)
        if M == 0:
            raise RuntimeError("Prompt pool is empty")
        top = np.argsort(-dist)[: min(self.sample_size, M)]
        ctxt = "\n\n".join(self.PC[i] for i in top)
        payload = f"{ctxt}\n\nQ: {test_q}\nA: Let's think step by step."
        seed = self.seed + len(self.train_questions) if self.seed is not None else None
        if semaphore:
            async with semaphore:
                return await self.llm.generate_async(payload, seed=seed)
        return await self.llm.generate_async(payload, seed=seed)

    __call__ = answer
