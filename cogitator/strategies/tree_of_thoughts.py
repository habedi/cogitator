"""Implements the Tree of Thoughts (ToT) framework."""

import asyncio
import logging
import math
import random
from typing import Any, AsyncIterator, Iterator, List, Optional, Tuple, Union

from ..model import BaseLLM
from ..schemas import EvaluationResult, ThoughtExpansion, Trace

logger = logging.getLogger(__name__)


class TreeOfThoughts:
    """Implements the Tree of Thoughts (ToT) prompting framework.

    ToT explores multiple reasoning paths concurrently in a tree structure.
    It uses an MCTS-like process involving selection (based on UCB1), expansion
    (generating potential next steps), evaluation (scoring paths), and backpropagation
    (updating node values) to guide the search towards promising reasoning paths.

    Reference:
        Yao et al. (v2; 2023) "Tree of Thoughts: Deliberate Problem Solving with Large Language Models".
        https://arxiv.org/abs/2305.10601
    """

    _node_id_counter: int = 0

    class _Node:
        """Represents a node in the Tree of Thoughts search tree."""

        __slots__ = ("children", "id", "parent", "prior", "steps", "value_sum", "visits")

        def __init__(
            self,
            steps: List[str],
            parent: Optional["TreeOfThoughts._Node"] = None,
            prior: float = 1.0,
        ) -> None:
            """Initializes a ToT node.

            Args:
                steps: List of reasoning steps leading to this node.
                parent: The parent node in the tree.
                prior: The prior probability/score for this node (used in UCB calculation).
            """
            self.id: int = TreeOfThoughts._node_id_counter
            TreeOfThoughts._node_id_counter += 1
            self.steps = steps
            self.parent = parent
            self.children: List["TreeOfThoughts._Node"] = []
            self.visits = 0
            self.value_sum = 0.0
            self.prior = prior

        def value(self) -> float:
            """Calculates the average value (score) of the node based on visits."""
            return self.value_sum / self.visits if self.visits > 0 else 0.0

        def is_leaf(self) -> bool:
            """Checks if the node is a leaf node (has no children)."""
            return not self.children

        def __repr__(self) -> str:
            """Returns a string representation of the node."""
            return (
                f"Node(id={self.id}, steps={len(self.steps)}, "
                f"val={self.value():.2f}, visits={self.visits})"
            )

    def __init__(
        self,
        llm: BaseLLM,
        max_depth: int = 3,
        num_branches: int = 5,
        sims: int = 16,
        c_puct: float = 1.0,
        expand_prompt: str = (
            "Generate {k} distinct reasoning steps or thoughts to continue solving the problem, given the context below. "
            "Return as a JSON object with a single key 'thoughts' containing a list of strings.\n\n"
            "Context:\n{ctx}\n"
            "Question: {question}\n\n"
            "JSON Output:"
        ),
        eval_prompt: str = (
            "Rate the quality of the reasoning steps below for solving the question on a scale of 1-10 "
            '(1=bad, 10=excellent). Return response as a JSON object with keys "score" (int) and "justification" (str).\n\n'
            "Question: {question}\n"
            "Steps:\n{steps}\n\n"
            "JSON Evaluation:"
        ),
        *,
        max_tokens: Optional[int] = 256,
        seed: Optional[int] = None,
    ) -> None:
        """Initializes the TreeOfThoughts strategy handler.

        Args:
            llm: The language model instance.
            max_depth: Maximum depth of the reasoning tree.
            num_branches: Number of thoughts to generate at each expansion step.
            sims: Number of MCTS simulations to run.
            c_puct: Exploration constant for the UCB1 formula in node selection.
            expand_prompt: Prompt template for the expansion step. Must include {k},
                           {ctx}, and {question}. Expects JSON output matching
                           ThoughtExpansion schema.
            eval_prompt: Prompt template for the evaluation step. Must include {question}
                         and {steps}. Expects JSON output matching EvaluationResult schema.
            max_tokens: Default maximum tokens for LLM generation calls.
            seed: Random seed for LLM calls.
        """
        self.llm = llm
        self.max_depth = max_depth
        self.num_branches = num_branches
        self.sims = sims
        self.c_puct = c_puct
        self.expand_prompt = expand_prompt
        self.eval_prompt = eval_prompt
        self.max_tokens = max_tokens
        self.seed = seed
        # Set random seed for reproducibility if provided
        if self.seed is not None:
            random.seed(self.seed)

    def _select(self, node: _Node, trace: Optional[Trace] = None) -> _Node:
        """Selects a leaf node for expansion using the UCB1 algorithm.

        Traverses the tree from the given node, at each step choosing the child
        with the highest UCB1 score until a leaf node is reached.

        Args:
            node: The starting node (usually the root).
            trace: Optional Trace object to record the selection path.

        Returns:
            The selected leaf node.
        """
        path = [node.id]
        while not node.is_leaf():
            total_visits = sum(child.visits for child in node.children)
            if total_visits == 0:
                # If no child has been visited, explore the first one first
                # or handle this case based on desired exploration strategy
                node = node.children[0] if node.children else node
                path.append(node.id)
                continue

            sqrt_total = math.sqrt(total_visits)
            ucb_scores = [
                child.value() + self.c_puct * child.prior * (sqrt_total / (1 + child.visits))
                for child in node.children
            ]
            # Find the index of the maximum UCB score
            # In case of ties, max() method's default behavior is used (typically returns the first max)
            best_idx = ucb_scores.index(max(ucb_scores))
            node = node.children[best_idx]
            path.append(node.id)

        if trace:
            # Add a metadata entry to the trace indicating the selection event
            # This is a conceptual example; trace structure may vary
            trace.add_node(
                node_id=f"select_{trace.root_node_id}_{len(trace.nodes)}",
                parent_id=path[-2] if len(path) > 1 else trace.root_node_id,
                content=f"Selected node {node.id} via path {path}",
                metadata={"type": "selection", "path": path, "final_node_id": node.id},
            )
        return node

    def _expand(
        self, node: _Node, question: str, trace: Optional[Trace] = None, **kwargs: Any
    ) -> None:
        """Expands a leaf node by generating child thoughts using the LLM.

        Uses `expand_prompt` to generate `num_branches` new thoughts based on
        the current node's steps. Creates child nodes for each valid thought.

        Args:
            node: The leaf node to expand.
            question: The original question being solved.
            trace: Optional Trace object to record new nodes.
            **kwargs: Additional arguments passed to the LLM `generate_json` call.
        """
        ctx = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(node.steps))
        prompt = self.expand_prompt.format(k=self.num_branches, ctx=ctx, question=question)
        try:
            local_kwargs = kwargs.copy()
            # Modify seed based on simulation/depth if needed for diversity
            expand_seed = local_kwargs.pop("seed", self.seed)
            if expand_seed is not None:
                # Example: vary seed based on node depth or simulation count if desired
                # expand_seed += len(node.steps)
                pass  # Keep base seed or implement variation logic

            generated = self.llm.generate_json(
                prompt,
                response_model=ThoughtExpansion,
                max_tokens=local_kwargs.pop("max_tokens", self.max_tokens),
                seed=expand_seed,
                **local_kwargs,
            )
            if not isinstance(generated, ThoughtExpansion):
                logger.warning("Expansion did not return ThoughtExpansion: %s", type(generated))
                return
            thoughts = [str(t).strip() for t in generated.thoughts if str(t).strip()]
        except Exception as e:
            logger.error("Expansion JSON failed: %s", e, exc_info=True)
            return

        prior = 1.0 / len(thoughts) if thoughts else 1.0
        for thought in thoughts[: self.num_branches]:
            child = TreeOfThoughts._Node([*node.steps, thought], parent=node, prior=prior)
            node.children.append(child)
            if trace:
                trace.add_node(
                    node_id=child.id,
                    parent_id=node.id,
                    content=thought,
                    metadata={"prior": prior, "type": "expansion"},
                )

    async def _expand_async(
        self,
        node: _Node,
        question: str,
        semaphore: Optional[asyncio.Semaphore],
        trace: Optional[Trace] = None,
        **kwargs: Any,
    ) -> None:
        """Asynchronously expands a leaf node.

        Similar to `_expand`, but uses async LLM calls.

        Args:
            node: The leaf node to expand.
            question: The original question being solved.
            semaphore: Optional asyncio.Semaphore to limit concurrent LLM calls.
            trace: Optional Trace object to record new nodes.
            **kwargs: Additional arguments passed to the async LLM `generate_json_async` call.
        """
        ctx = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(node.steps))
        prompt = self.expand_prompt.format(k=self.num_branches, ctx=ctx, question=question)
        try:
            local_kwargs = kwargs.copy()
            expand_seed = local_kwargs.pop("seed", self.seed)
            gen_args = {
                "response_model": ThoughtExpansion,
                "max_tokens": local_kwargs.pop("max_tokens", self.max_tokens),
                "seed": expand_seed,
                **local_kwargs,
            }
            if semaphore:
                async with semaphore:
                    generated = await self.llm.generate_json_async(prompt, **gen_args)
            else:
                generated = await self.llm.generate_json_async(prompt, **gen_args)

            if not isinstance(generated, ThoughtExpansion):
                logger.warning(
                    "Async expansion did not return ThoughtExpansion: %s", type(generated)
                )
                return
            thoughts = [str(t).strip() for t in generated.thoughts if str(t).strip()]
        except Exception as e:
            logger.error("Async expansion JSON failed: %s", e, exc_info=True)
            return

        prior = 1.0 / len(thoughts) if thoughts else 1.0
        for thought in thoughts[: self.num_branches]:
            child = TreeOfThoughts._Node([*node.steps, thought], parent=node, prior=prior)
            node.children.append(child)
            if trace:
                trace.add_node(
                    node_id=child.id,
                    parent_id=node.id,
                    content=thought,
                    metadata={"prior": prior, "type": "expansion"},
                )

    def _evaluate(
        self, node: _Node, question: str, trace: Optional[Trace] = None, **kwargs: Any
    ) -> float:
        """Evaluates the reasoning path leading to a node using the LLM.

        Uses `eval_prompt` and expects a JSON response matching `EvaluationResult`.
        Normalizes the score to be between 0.0 and 1.0.

        Args:
            node: The node whose path is to be evaluated.
            question: The original question.
            trace: Optional Trace object to record the evaluation.
            **kwargs: Additional arguments passed to the LLM `generate_json` call.

        Returns:
            The normalized evaluation score (0.0 to 1.0), or 0.0 on failure.
        """
        steps_str = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(node.steps))
        prompt = self.eval_prompt.format(question=question, steps=steps_str)
        score = 0.0
        justification = "Evaluation failed"
        try:
            local_kwargs = kwargs.copy()
            eval_seed = local_kwargs.pop("seed", self.seed)
            # Optional seed variation logic here
            result = self.llm.generate_json(
                prompt,
                response_model=EvaluationResult,
                max_tokens=local_kwargs.pop("max_tokens", self.max_tokens),
                seed=eval_seed,
                **local_kwargs,
            )
            if isinstance(result, EvaluationResult):
                raw = float(result.score)
                score = max(0.0, min(1.0, (raw - 1.0) / 9.0))
                justification = result.justification
        except Exception as e:
            logger.error("Eval JSON failed: %s", e, exc_info=True)

        if trace:
            # Find the corresponding node in the trace and update it
            for trace_node in trace.nodes:
                if trace_node.node_id == node.id:
                    trace_node.score = score
                    if trace_node.metadata:
                        trace_node.metadata["justification"] = justification
                    else:
                        trace_node.metadata = {"justification": justification}
                    break
        return score

    async def _evaluate_async(
        self,
        node: _Node,
        question: str,
        semaphore: Optional[asyncio.Semaphore],
        trace: Optional[Trace] = None,
        **kwargs: Any,
    ) -> float:
        """Asynchronously evaluates the reasoning path leading to a node.

        Similar to `_evaluate`, but uses async LLM calls.

        Args:
            node: The node whose path is to be evaluated.
            question: The original question.
            semaphore: Optional asyncio.Semaphore to limit concurrent LLM calls.
            trace: Optional Trace object to record the evaluation.
            **kwargs: Additional arguments passed to the async LLM `generate_json_async` call.

        Returns:
            The normalized evaluation score (0.0 to 1.0), or 0.0 on failure.
        """
        steps_str = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(node.steps))
        prompt = self.eval_prompt.format(question=question, steps=steps_str)
        score = 0.0
        justification = "Evaluation failed"
        try:
            local_kwargs = kwargs.copy()
            eval_seed = local_kwargs.pop("seed", self.seed)
            gen_args = {
                "response_model": EvaluationResult,
                "max_tokens": local_kwargs.pop("max_tokens", self.max_tokens),
                "seed": eval_seed,
                **local_kwargs,
            }
            if semaphore:
                async with semaphore:
                    result = await self.llm.generate_json_async(prompt, **gen_args)
            else:
                result = await self.llm.generate_json_async(prompt, **gen_args)

            if isinstance(result, EvaluationResult):
                raw = float(result.score)
                score = max(0.0, min(1.0, (raw - 1.0) / 9.0))
                justification = result.justification
        except Exception as e:
            logger.error("Async eval JSON failed: %s", e, exc_info=True)

        if trace:
            for trace_node in trace.nodes:
                if trace_node.node_id == node.id:
                    trace_node.score = score
                    if trace_node.metadata:
                        trace_node.metadata["justification"] = justification
                    else:
                        trace_node.metadata = {"justification": justification}
                    break
        return score

    def _backpropagate(self, node: _Node, value: float, trace: Optional[Trace] = None) -> None:
        """Backpropagates the evaluation score up the tree.

        Increments the visit count and adds the value to the value sum for the
        given node and all its ancestors up to the root.

        Args:
            node: The node from which to start backpropagation.
            value: The evaluation score to propagate.
            trace: Optional Trace object to record the updates.
        """
        cur = node
        while cur is not None:
            cur.visits += 1
            cur.value_sum += value
            if trace:
                for trace_node in trace.nodes:
                    if trace_node.node_id == cur.id:
                        trace_node.visits = cur.visits
                        trace_node.score = cur.value()  # Update with the new average value
                        break
            cur = cur.parent

    def run(
        self, question: str, with_trace: bool = False, **kwargs: Any
    ) -> Union[str, Tuple[str, Trace]]:
        """Executes the Tree of Thoughts search process using MCTS-like steps.

        Performs `sims` simulations. Each simulation involves:
        1. Selection: Traverse the tree using UCB1 to find a promising leaf node.
        2. Expansion: If the leaf is not at max depth, generate potential next thoughts.
        3. Evaluation: Evaluate a randomly chosen newly generated child node (if expansion occurred)
           or the selected leaf node itself (if terminal or expansion failed).
        4. Backpropagation: Update visit counts and value sums up the tree from the evaluated node.

        After simulations, selects the most promising path (based on visits and value)
        and generates the final answer using the steps from that path as context.

        Args:
            question: The question to solve.
            with_trace: If True, returns a tuple of (answer, trace).
            **kwargs: Additional arguments passed to internal LLM calls.

        Returns:
            The final answer string, or a tuple (answer, trace) if with_trace is True.
        """
        TreeOfThoughts._node_id_counter = 0
        root = TreeOfThoughts._Node(steps=[], parent=None, prior=1.0)
        trace: Optional[Trace] = None
        if with_trace:
            trace = Trace(root_node_id=root.id)
            trace.add_node(
                node_id=root.id,
                parent_id=None,
                content="<root>",
                score=root.value(),
                visits=root.visits,
            )

        for sim in range(self.sims):
            logger.debug("Simulation %d/%d", sim + 1, self.sims)
            leaf = self._select(root, trace=trace)

            to_eval = leaf
            if len(leaf.steps) < self.max_depth:
                self._expand(leaf, question, trace=trace, **kwargs)
                if leaf.children:
                    # Evaluate a randomly chosen new child node
                    to_eval = random.choice(leaf.children)

            value = self._evaluate(to_eval, question, trace=trace, **kwargs)
            self._backpropagate(to_eval, value, trace=trace)

        if not root.children:
            logger.warning("No thoughts were generated; answering directly.")
            final_prompt = f"Answer the question: {question}"
        else:
            node = root
            while node.children:
                best_child = max(node.children, key=lambda c: (c.visits, c.value()))
                node = best_child
            ctx = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(node.steps))
            final_prompt = f"Given reasoning steps:\n{ctx}\n\nAnswer the question: {question}"

        try:
            local_kwargs = kwargs.copy()
            final_seed = local_kwargs.pop("seed", self.seed)
            # Optional: Use a different seed for the final generation
            answer = self.llm.generate(
                final_prompt,
                max_tokens=local_kwargs.pop("max_tokens", self.max_tokens),
                seed=final_seed,
                **local_kwargs,
            ).strip()
            return (answer, trace) if with_trace and trace else answer
        except Exception as e:
            logger.error("Final answer generation failed: %s", e, exc_info=True)
            err_msg = "Error generating final answer."
            return (err_msg, trace) if with_trace and trace else err_msg

    async def run_async(
        self,
        question: str,
        semaphore: Optional[asyncio.Semaphore] = None,
        with_trace: bool = False,
        **kwargs: Any,
    ) -> Union[str, Tuple[str, Trace]]:
        """Asynchronously executes the Tree of Thoughts search process using MCTS-like steps.

        Similar to `run`, but performs expansion and evaluation steps concurrently
        using asyncio. Evaluates a randomly chosen child after expansion.

        Args:
            question: The question to solve.
            semaphore: Optional asyncio.Semaphore to limit concurrent LLM calls.
            with_trace: If True, returns a tuple of (answer, trace).
            **kwargs: Additional arguments passed to internal async LLM calls.

        Returns:
            The final answer string, or a tuple (answer, trace) if with_trace is True.
        """
        TreeOfThoughts._node_id_counter = 0
        root = TreeOfThoughts._Node(steps=[], parent=None, prior=1.0)
        trace: Optional[Trace] = None
        if with_trace:
            trace = Trace(root_node_id=root.id)
            trace.add_node(
                node_id=root.id,
                parent_id=None,
                content="<root>",
                score=root.value(),
                visits=root.visits,
            )

        for sim in range(self.sims):
            logger.debug("Async Simulation %d/%d", sim + 1, self.sims)
            leaf = self._select(root, trace=trace)

            to_eval = leaf
            if len(leaf.steps) < self.max_depth:
                await self._expand_async(leaf, question, semaphore, trace=trace, **kwargs)
                if leaf.children:
                    # Evaluate a randomly chosen new child node
                    to_eval = random.choice(leaf.children)

            value = await self._evaluate_async(to_eval, question, semaphore, trace=trace, **kwargs)
            self._backpropagate(to_eval, value, trace=trace)

        if not root.children:
            logger.warning("No thoughts were generated async; answering directly.")
            final_prompt = f"Answer the question: {question}"
        else:
            node = root
            while node.children:
                best_child = max(node.children, key=lambda c: (c.visits, c.value()))
                node = best_child
            ctx = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(node.steps))
            final_prompt = f"Given reasoning steps:\n{ctx}\n\nAnswer the question: {question}"

        try:
            local_kwargs = kwargs.copy()
            final_seed = local_kwargs.pop("seed", self.seed)
            # Optional: Use a different seed for the final generation
            gen_args = {
                "max_tokens": local_kwargs.pop("max_tokens", self.max_tokens),
                "seed": final_seed,
                **local_kwargs,
            }
            if semaphore:
                async with semaphore:
                    answer = (await self.llm.generate_async(final_prompt, **gen_args)).strip()
            else:
                answer = (await self.llm.generate_async(final_prompt, **gen_args)).strip()

            return (answer, trace) if with_trace and trace else answer
        except Exception as e:
            logger.error("Final async answer generation failed: %s", e, exc_info=True)
            err_msg = "Error generating final async answer."
            return (err_msg, trace) if with_trace and trace else err_msg

    # Consider adding stream methods if applicable, though ToT structure makes streaming complex.
    def run_stream(self, prompt: str) -> Iterator[str]:
        """Streaming is not directly supported by the standard ToT search."""
        raise NotImplementedError("Streaming not supported for TreeOfThoughts.")

    async def run_stream_async(self, prompt: str) -> AsyncIterator[str]:
        """Streaming is not directly supported by the standard ToT search."""
        raise NotImplementedError("Streaming not supported for TreeOfThoughts.")
