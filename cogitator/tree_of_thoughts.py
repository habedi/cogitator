import asyncio
import logging
import math
from typing import List, Optional

from .model import BaseLLM
from .schemas import EvaluationResult, ThoughtExpansion

logger = logging.getLogger(__name__)


class TreeOfThoughts:
    class _Node:
        __slots__ = ("steps", "parent", "children", "visits", "value_sum", "prior")

        def __init__(
            self,
            steps: List[str],
            parent: Optional["TreeOfThoughts._Node"] = None,
            prior: float = 1.0,
        ):
            self.steps = steps
            self.parent = parent
            self.children: List["TreeOfThoughts._Node"] = []
            self.visits = 0
            self.value_sum = 0.0
            self.prior = prior

        def value(self) -> float:
            return self.value_sum / self.visits if self.visits > 0 else 0.0

        def is_leaf(self) -> bool:
            return not self.children

        def __repr__(self) -> str:
            return f"Node(steps={len(self.steps)}, val={self.value():.2f}, visits={self.visits})"

    def __init__(
        self,
        llm: BaseLLM,
        max_depth: int = 3,
        num_branches: int = 5,
        sims: int = 16,
        c_puct: float = 1.0,
        expand_prompt: str = (
            "Generate {k} distinct reasoning steps to continue solving the problem, given the context below. "
            "Return as a JSON list of strings.\n\n"
            "Context:\n{ctx}\n"
            "Question: {question}\n\n"
            "JSON Steps:"
        ),
        eval_prompt: str = (
            "Rate the quality of the reasoning steps below for solving the question on a scale of 1-10 "
            '(1=bad, 10=excellent). Return response as a JSON object with keys "score" (int) and "justification" (str).\n\n'
            "Question: {question}\n"
            "Steps:\n{steps}\n\n"
            "JSON Evaluation:"
        ),
    ):
        self.llm = llm
        self.max_depth = max_depth
        self.num_branches = num_branches
        self.sims = sims
        self.c_puct = c_puct
        self.expand_prompt = expand_prompt
        self.eval_prompt = eval_prompt

    def _select(self, node: _Node) -> _Node:
        while not node.is_leaf():
            total_visits = sum(child.visits for child in node.children)
            if total_visits == 0:
                if not node.children:
                    break
                else:
                    node = node.children[0]
                    continue

            sqrt_total = math.sqrt(total_visits)

            ucb_scores = [
                child.value() + self.c_puct * child.prior * (sqrt_total / (1 + child.visits))
                for child in node.children
            ]
            if not ucb_scores:
                break
            best_idx = ucb_scores.index(max(ucb_scores))
            node = node.children[best_idx]
        return node

    def _expand(self, node: _Node, question: str) -> None:
        ctx = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(node.steps))
        prompt = self.expand_prompt.format(k=self.num_branches, ctx=ctx, question=question)
        generated = None
        try:
            generated = self.llm.generate_json(prompt, response_model=ThoughtExpansion)
            # Check type based on Pydantic validation
            if not isinstance(generated, ThoughtExpansion):
                logger.warning("Expansion did not return ThoughtExpansion: %s", type(generated))
                return
            expansion_list = generated.thoughts
        except Exception as e:
            logger.error("Expansion JSON failed: %s", e, exc_info=True)
            return

        thoughts = [str(t).strip() for t in expansion_list if str(t).strip()]
        if not thoughts:
            logger.debug("No valid thoughts extracted from expansion: %s", expansion_list)
            return
        prior = 1.0 / len(thoughts) if thoughts else 1.0
        for thought in thoughts[: self.num_branches]:
            child = TreeOfThoughts._Node(node.steps + [thought], parent=node, prior=prior)
            node.children.append(child)

    async def _expand_async(
        self, node: _Node, question: str, semaphore: Optional[asyncio.Semaphore]
    ) -> None:
        ctx = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(node.steps))
        prompt = self.expand_prompt.format(k=self.num_branches, ctx=ctx, question=question)
        generated = None
        try:
            if semaphore:
                async with semaphore:
                    generated = await self.llm.generate_json_async(
                        prompt, response_model=ThoughtExpansion
                    )
            else:
                generated = await self.llm.generate_json_async(
                    prompt, response_model=ThoughtExpansion
                )

            if not isinstance(generated, ThoughtExpansion):
                logger.warning(
                    "Async expansion did not return ThoughtExpansion: %s", type(generated)
                )
                return
            expansion_list = generated.thoughts
        except Exception as e:
            logger.error("Async expansion JSON failed: %s", e, exc_info=True)
            return

        thoughts = [str(t).strip() for t in expansion_list if str(t).strip()]
        if not thoughts:
            logger.debug("No valid thoughts extracted from async expansion: %s", expansion_list)
            return
        prior = 1.0 / len(thoughts) if thoughts else 1.0
        for thought in thoughts[: self.num_branches]:
            child = TreeOfThoughts._Node(node.steps + [thought], parent=node, prior=prior)
            node.children.append(child)

    def _evaluate(self, node: _Node, question: str) -> float:
        steps_str = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(node.steps))
        prompt = self.eval_prompt.format(question=question, steps=steps_str)
        result = None
        try:
            result = self.llm.generate_json(prompt, response_model=EvaluationResult)
            if isinstance(result, EvaluationResult):
                raw = float(result.score)
                return max(0.0, min(1.0, (raw - 1.0) / 9.0))
            logger.warning("Eval did not return expected EvaluationResult type: %s", type(result))
        except (ValueError, TypeError) as e:
            result_str = repr(result) if result is not None else "[result unavailable]"
            logger.error(
                "Error converting evaluation score to float: %s. Result was: %s", e, result_str
            )
        except Exception as e:
            logger.error("Eval JSON generation/parsing failed: %s", e, exc_info=True)
        return 0.0

    async def _evaluate_async(
        self, node: _Node, question: str, semaphore: Optional[asyncio.Semaphore]
    ) -> float:
        steps_str = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(node.steps))
        prompt = self.eval_prompt.format(question=question, steps=steps_str)
        result = None
        try:
            if semaphore:
                async with semaphore:
                    result = await self.llm.generate_json_async(
                        prompt, response_model=EvaluationResult
                    )
            else:
                result = await self.llm.generate_json_async(prompt, response_model=EvaluationResult)

            if isinstance(result, EvaluationResult):
                raw = float(result.score)
                return max(0.0, min(1.0, (raw - 1.0) / 9.0))
            logger.warning(
                "Async eval did not return expected EvaluationResult type: %s", type(result)
            )
        except (ValueError, TypeError) as e:
            result_str = repr(result) if result is not None else "[result unavailable]"
            logger.error(
                "Error converting async evaluation score to float: %s. Result was: %s",
                e,
                result_str,
            )
        except Exception as e:
            logger.error("Async eval JSON generation/parsing failed: %s", e, exc_info=True)
        return 0.0

    def _backpropagate(self, node: _Node, value: float) -> None:
        cur = node
        while cur is not None:
            cur.visits += 1
            cur.value_sum += value
            cur = cur.parent

    def run(self, question: str) -> str:
        root = TreeOfThoughts._Node(steps=[], parent=None, prior=1.0)

        for sim in range(self.sims):
            logger.debug("Simulation %d/%d", sim + 1, self.sims)
            leaf = self._select(root)

            to_eval = leaf
            if len(leaf.steps) < self.max_depth:
                self._expand(leaf, question)
                if leaf.children:
                    to_eval = leaf.children[0]

            value = self._evaluate(to_eval, question)
            self._backpropagate(to_eval, value)

        if not root.children:
            logger.warning("No thoughts were generated; answering directly.")
            final_prompt = f"Answer the question: {question}"
        else:
            node = root
            path = [node]
            while node.children:
                best_child = max(node.children, key=lambda c: (c.visits, c.value()))
                if best_child.visits == 0 and all(c.visits == 0 for c in node.children):
                    best_child = node.children[0]
                    logger.debug("No children visited, selecting first available child")
                node = best_child
                path.append(node)

            best_final_node = path[-1]
            ctx = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(best_final_node.steps))
            final_prompt = f"Given reasoning steps:\n{ctx}\n\nAnswer the question: {question}"

        try:
            return self.llm.generate(final_prompt).strip()
        except Exception as e:
            logger.error("Final answer generation failed: %s", e, exc_info=True)
            return "Error generating final answer."

    async def run_async(self, question: str, semaphore: Optional[asyncio.Semaphore] = None) -> str:
        root = TreeOfThoughts._Node(steps=[], parent=None, prior=1.0)

        for sim in range(self.sims):
            logger.debug("Async Simulation %d/%d", sim + 1, self.sims)
            leaf = self._select(root)

            eval_node = leaf
            if len(leaf.steps) < self.max_depth:
                await self._expand_async(leaf, question, semaphore)
                if leaf.children:
                    eval_node = leaf.children[0]

            value = await self._evaluate_async(eval_node, question, semaphore)
            self._backpropagate(eval_node, value)

        if not root.children:
            logger.warning("No thoughts were generated async; answering directly.")
            final_prompt = f"Answer the question: {question}"
        else:
            node = root
            path = [node]
            while node.children:
                best_child = max(node.children, key=lambda c: (c.visits, c.value()))
                if best_child.visits == 0 and all(c.visits == 0 for c in node.children):
                    best_child = node.children[0]
                    logger.debug("Async: No children visited, selecting first available child")
                node = best_child
                path.append(node)

            best_final_node = path[-1]
            ctx = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(best_final_node.steps))
            final_prompt = f"Given reasoning steps:\n{ctx}\n\nAnswer the question: {question}"

        try:
            if semaphore:
                async with semaphore:
                    return (await self.llm.generate_async(final_prompt)).strip()
            else:
                return (await self.llm.generate_async(final_prompt)).strip()
        except Exception as e:
            logger.error("Final async answer generation failed: %s", e, exc_info=True)
            return "Error generating final async answer."

    __call__ = run
