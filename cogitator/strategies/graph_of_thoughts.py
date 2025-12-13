"""Implements the Graph of Thoughts (GoT) framework."""

import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from ..embedding import BaseEmbedder, SentenceTransformerEmbedder
from ..model import BaseLLM
from ..schemas import EvaluationResult, ExtractedAnswer, ThoughtExpansion, Trace

logger = logging.getLogger(__name__)


def _strip_fences(text: str) -> str:
    """Removes Markdown code fences (```json ... ``` or ``` ... ```) from text."""
    t = text.strip()
    match = re.match(r"```(?:json)?\s*(.*)\s*```", t, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    if t.startswith("```") and t.endswith("```"):
        return t[3:-3].strip()
    return t


class GoTNode:
    """Represents a node (thought) in the Graph of Thoughts."""

    __slots__ = (
        "children",
        "data",
        "embed",
        "id",
        "parents",
        "score",
        "steps",
        "text_content",
        "valid",
        "visits",
    )
    _id_counter = 0

    def __init__(
        self,
        steps: List[str],
        embedder: Optional[BaseEmbedder] = None,
        parents: Optional[List["GoTNode"]] = None,
        data: Optional[Any] = None,
        text_content: Optional[str] = None,  # Store the core text content separately if needed
    ) -> None:
        """Initializes a GoT node.

        Args:
            steps: The sequence of reasoning steps/operations leading to this node.
            embedder: The embedding model used for calculating node similarity. Can be None.
            parents: A list of parent nodes.
            data: Optional arbitrary data associated with the node.
            text_content: Optional string representing the actual thought content.
        """
        self.id: int = GoTNode._id_counter
        GoTNode._id_counter += 1

        self.steps: List[str] = steps  # History or description of how node was created
        self.text_content: Optional[str] = text_content  # The actual thought text
        self.parents: List["GoTNode"] = parents or []
        self.children: List["GoTNode"] = []
        self.embed: Optional[np.ndarray] = None
        self.visits: int = 0  # Can be used for MCTS-like scores or just tracking
        self.score: float = 0.0  # Store the latest evaluated score directly
        self.valid: Optional[bool] = None  # Validity status
        self.data: Optional[Any] = data  # Store auxiliary data

        if embedder and self.text_content:
            try:
                emb_list = embedder.encode([self.text_content])
                if len(emb_list) > 0 and emb_list[0] is not None:
                    self.embed = np.array(emb_list[0], dtype=float)
            except Exception as e:
                logger.error("Failed to encode node %d content: %s", self.id, e)
                self.embed = None

    def is_ancestor(self, potential_ancestor: "GoTNode") -> bool:
        """Checks if `potential_ancestor` is an ancestor of this node."""
        if not self.parents:
            return False
        queue = list(self.parents)
        visited = {self.id}
        while queue:
            p = queue.pop(0)
            if p.id == potential_ancestor.id:
                return True
            if p.id not in visited:
                visited.add(p.id)
                queue.extend(p.parents)
        return False

    def __repr__(self) -> str:
        """Returns a string representation of the node."""
        pids = [p.id for p in self.parents]
        content_preview = f"'{self.text_content[:20]}...'" if self.text_content else "None"
        return (
            f"Node(id={self.id}, score={self.score:.2f}, valid={self.valid}, "
            f"parents={pids}, content={content_preview})"
        )


class GraphReasoningState:
    """Maintains the dynamic state of the GoT reasoning process."""

    def __init__(self, root_node: GoTNode) -> None:
        """Initializes the Graph Reasoning State.

        Args:
            root_node: The initial node containing the problem input.
        """
        self.all_nodes: Dict[int, GoTNode] = {root_node.id: root_node}
        # 'active_sets' can store different groups of nodes, e.g., 'frontier', 'scored', 'generated_in_step_X'
        self.active_sets: Dict[str, List[GoTNode]] = {"frontier": [root_node]}

    def add_node(self, node: GoTNode) -> None:
        """Adds a new node to the state."""
        if node.id not in self.all_nodes:
            self.all_nodes[node.id] = node
            for parent in node.parents:
                if parent.id in self.all_nodes:
                    self.all_nodes[parent.id].children.append(node)
        else:
            logger.warning(f"Attempted to add duplicate node ID {node.id}")

    def get_nodes(self, node_ids: List[int]) -> List[GoTNode]:
        """Retrieves nodes by their IDs."""
        return [self.all_nodes[nid] for nid in node_ids if nid in self.all_nodes]

    def get_active_set(self, set_name: str) -> List[GoTNode]:
        """Gets the list of nodes in a named active set."""
        return self.active_sets.get(set_name, [])

    def set_active_set(self, set_name: str, nodes: List[GoTNode]) -> None:
        """Sets or replaces a named active set."""
        self.active_sets[set_name] = nodes

    def update_node(self, node: GoTNode) -> None:
        """Updates an existing node in the state (e.g., score, validity)."""
        if node.id in self.all_nodes:
            # Update fields as needed, e.g.:
            self.all_nodes[node.id].score = node.score
            self.all_nodes[node.id].valid = node.valid
            # Be careful not to overwrite structural info like parents/children unless intended
        else:
            logger.warning(f"Attempted to update non-existent node ID {node.id}")


# --- Operation Base Class ---
class GoTOperation(ABC):
    """Abstract base class for all Graph of Thoughts operations."""

    def __init__(self, **params) -> None:
        """Initializes the operation with specific parameters."""
        self.params = params

    @abstractmethod
    def execute(
        self,
        grs: GraphReasoningState,
        llm: BaseLLM,
        prompts: Dict[str, str],
        embedder: Optional[BaseEmbedder] = None,
        **global_kwargs: Any,
    ) -> None:
        """Executes the operation, modifying the GraphReasoningState.

        Args:
            grs: The current graph reasoning state.
            llm: The language model instance.
            prompts: A dictionary of available prompt templates.
            embedder: Optional embedder for operations needing similarity.
            global_kwargs: Global arguments like seed, max_tokens passed to LLM calls.
        """
        pass

    # Optional: async version
    @abstractmethod
    async def execute_async(
        self,
        grs: GraphReasoningState,
        llm: BaseLLM,
        prompts: Dict[str, str],
        embedder: Optional[BaseEmbedder] = None,
        semaphore: Optional[asyncio.Semaphore] = None,
        trace: Optional[Trace] = None,
        **global_kwargs,
    ) -> None:
        """Asynchronously executes the operation."""
        pass


# --- Concrete Operation Implementations (Examples) ---


class GenerateOp(GoTOperation):
    """Generates new thoughts based on parent nodes."""

    async def execute_async(
        self,
        grs: GraphReasoningState,
        llm: BaseLLM,
        prompts: Dict[str, str],
        embedder: Optional[BaseEmbedder] = None,
        semaphore: Optional[asyncio.Semaphore] = None,
        trace: Optional[Trace] = None,
        **global_kwargs: Any,
    ) -> None:
        """Generates new thoughts asynchronously."""
        k = self.params.get("k", 1)
        target_set = self.params.get("target_set", "frontier")
        output_set = self.params.get(
            "output_set", "generated"
        )  # Where to store newly generated nodes
        prompt_key = self.params.get("prompt_key", "expand")  # e.g., 'sort', 'expand'
        response_schema = self.params.get("response_schema", ThoughtExpansion)  # Optional schema

        parent_nodes = grs.get_active_set(target_set)
        newly_generated_nodes = []

        async def gen_task(parent_node):
            ctx = parent_node.text_content or "\n".join(parent_node.steps)  # Get context
            prompt = prompts[prompt_key].format(
                k=k, ctx=ctx, **parent_node.data if parent_node.data else {}
            )

            try:
                local_kwargs = global_kwargs.copy()
                gen_args = {
                    "max_tokens": local_kwargs.pop("max_tokens", None),
                    "seed": local_kwargs.pop("seed", None),  # Add seed variation logic if needed
                    **local_kwargs,
                }
                # Use generate_json if schema provided, else generate
                if response_schema:
                    gen_args["response_model"] = response_schema
                    if semaphore:
                        async with semaphore:
                            result = await llm.generate_json_async(prompt, **gen_args)
                    else:
                        result = await llm.generate_json_async(prompt, **gen_args)

                    # Adapt parsing based on schema (e.g., ThoughtExpansion)
                    thoughts_texts = [
                        str(t).strip() for t in getattr(result, "thoughts", []) if str(t).strip()
                    ]

                else:  # Assuming the response is a list of strings in JSON
                    if semaphore:
                        async with semaphore:
                            raw = await llm.generate_async(prompt, **gen_args)
                    else:
                        raw = await llm.generate_async(prompt, **gen_args)

                    # Simplified parsing - adjust as needed
                    try:
                        parsed_list = json.loads(_strip_fences(raw))
                        thoughts_texts = [
                            str(t).strip()
                            for t in parsed_list
                            if isinstance(t, (str, int, float)) and str(t).strip()
                        ]
                    except Exception:
                        thoughts_texts = []
                        logger.error(
                            f"Failed to parse GenerateOp JSON response for node {parent_node.id}"
                        )

                node_results = []
                for thought_text in thoughts_texts[:k]:
                    new_node = GoTNode(
                        steps=[*parent_node.steps, f"Generate({prompt_key})"],
                        parents=[parent_node],
                        text_content=thought_text,
                        embedder=embedder,
                    )
                    grs.add_node(
                        new_node
                    )  # Add immediately to allow potential merging later if needed
                    if trace:
                        trace.add_node(
                            node_id=new_node.id,
                            parent_id=parent_node.id,
                            content=new_node.text_content or "",
                            metadata={"op": "Generate", "prompt_key": prompt_key},
                        )
                    node_results.append(new_node)
                return node_results
            except Exception as e:
                logger.error(
                    f"Generate task failed for parent {parent_node.id}: {e}", exc_info=True
                )
                return []

        results = await asyncio.gather(*(gen_task(node) for node in parent_nodes))
        for node_list in results:
            newly_generated_nodes.extend(node_list)

        grs.set_active_set(output_set, newly_generated_nodes)
        logger.info(
            f"GenerateOp created {len(newly_generated_nodes)} new nodes in set '{output_set}'."
        )

    def execute(
        self,
        grs: GraphReasoningState,
        llm: BaseLLM,
        prompts: Dict[str, str],
        embedder: Optional[BaseEmbedder] = None,
        **global_kwargs: Any,
    ) -> None:
        # Synchronous version would mirror async logic without async/await/gather
        raise NotImplementedError(
            "Synchronous execute not fully implemented for GenerateOp sketch."
        )


class AggregateOp(GoTOperation):
    """Aggregates multiple thoughts into new ones."""

    async def execute_async(
        self,
        grs: GraphReasoningState,
        llm: BaseLLM,
        prompts: Dict[str, str],
        embedder: Optional[BaseEmbedder] = None,
        semaphore: Optional[asyncio.Semaphore] = None,
        trace: Optional[Trace] = None,
        **global_kwargs: Any,
    ) -> None:
        """Aggregates thoughts asynchronously."""
        k = self.params.get("k", 1)
        target_sets = self.params.get(
            "target_sets", ["frontier"]
        )  # List of set names to aggregate from
        output_set = self.params.get("output_set", "aggregated")
        prompt_key = self.params.get("prompt_key", "aggregate")  # e.g., 'merge', 'summarize'

        nodes_to_aggregate = []
        for set_name in target_sets:
            nodes_to_aggregate.extend(grs.get_active_set(set_name))

        if not nodes_to_aggregate:
            logger.warning("AggregateOp: No nodes found in target sets.")
            grs.set_active_set(output_set, [])
            return

        contexts = []
        parent_refs = []
        for i, node in enumerate(nodes_to_aggregate):
            contexts.append(f"Input {i + 1}:\n{node.text_content or 'No Content'}\n")
            parent_refs.append(node)
        full_context = "\n".join(contexts)

        prompt = prompts[prompt_key].format(k=k, context=full_context)
        newly_aggregated_nodes = []

        try:
            local_kwargs = global_kwargs.copy()
            gen_args = {
                "max_tokens": local_kwargs.pop("max_tokens", None),
                "seed": local_kwargs.pop("seed", None),
                **local_kwargs,
            }

            if semaphore:
                async with semaphore:
                    raw = await llm.generate_async(prompt, **gen_args)
            else:
                raw = await llm.generate_async(prompt, **gen_args)

            # Simplified parsing - assuming a list of aggregated thoughts
            try:
                parsed_list = json.loads(_strip_fences(raw))
                aggregated_texts = [
                    str(t).strip()
                    for t in parsed_list
                    if isinstance(t, (str, int, float)) and str(t).strip()
                ]
            except Exception:
                aggregated_texts = [
                    _strip_fences(raw)
                ]  # Fallback: treat the whole output as one thought
                if not aggregated_texts[0]:
                    logger.error("Failed to parse AggregateOp JSON response.")
                    aggregated_texts = []

            for agg_text in aggregated_texts[:k]:
                new_node = GoTNode(
                    steps=[f"Aggregate({prompt_key}) from {[p.id for p in parent_refs]}"],
                    parents=parent_refs,
                    text_content=agg_text,
                    embedder=embedder,
                )
                grs.add_node(new_node)
                if trace:
                    trace.add_node(
                        node_id=new_node.id,
                        # This part is tricky as there are multiple parents.
                        # We can list them in metadata.
                        parent_id=parent_refs[0].id if parent_refs else None,
                        content=new_node.text_content or "",
                        metadata={
                            "op": "Aggregate",
                            "prompt_key": prompt_key,
                            "parent_ids": [p.id for p in parent_refs],
                        },
                    )
                newly_aggregated_nodes.append(new_node)

        except Exception as e:
            logger.error(
                f"Aggregate task failed for parents {[p.id for p in parent_refs]}: {e}",
                exc_info=True,
            )

        grs.set_active_set(output_set, newly_aggregated_nodes)
        logger.info(
            f"AggregateOp created {len(newly_aggregated_nodes)} new nodes in set '{output_set}'."
        )

    def execute(
        self,
        grs: GraphReasoningState,
        llm: BaseLLM,
        prompts: Dict[str, str],
        embedder: Optional[BaseEmbedder] = None,
        **global_kwargs: Any,
    ) -> None:
        raise NotImplementedError(
            "Synchronous execute not fully implemented for AggregateOp sketch."
        )


class ScoreOp(GoTOperation):
    """Scores thoughts using the LLM."""

    async def execute_async(
        self,
        grs: GraphReasoningState,
        llm: BaseLLM,
        prompts: Dict[str, str],
        embedder: Optional[BaseEmbedder] = None,
        semaphore: Optional[asyncio.Semaphore] = None,
        trace: Optional[Trace] = None,
        **global_kwargs: Any,
    ) -> None:
        """Scores nodes asynchronously."""
        target_set = self.params.get(
            "target_set", "generated"
        )  # Score nodes generated in a previous step
        prompt_key = self.params.get("prompt_key", "evaluate")

        nodes_to_score = grs.get_active_set(target_set)

        async def score_task(node):
            if not node.text_content:
                return node.id, 0.0  # Cannot score nodes without content

            steps_str = node.text_content  # Or format node.steps if needed
            prompt = prompts[prompt_key].format(
                steps=steps_str
            )  # Prompt needs 'steps' or similar key

            try:
                local_kwargs = global_kwargs.copy()
                gen_args = {
                    "response_model": EvaluationResult,
                    "max_tokens": local_kwargs.pop("max_tokens", None),
                    "seed": local_kwargs.pop("seed", None),  # Add seed variation
                    **local_kwargs,
                }
                if semaphore:
                    async with semaphore:
                        result = await llm.generate_json_async(prompt, **gen_args)
                else:
                    result = await llm.generate_json_async(prompt, **gen_args)

                if isinstance(result, EvaluationResult):
                    raw = float(result.score)
                    normalized_score = max(0.0, min(1.0, (raw - 1.0) / 9.0))
                    return node.id, normalized_score
                else:
                    logger.error(
                        f"ScoreOp returned unexpected type for node {node.id}: {type(result)}"
                    )
                    return node.id, 0.0
            except Exception as e:
                logger.error(f"Score task failed for node {node.id}: {e}", exc_info=True)
                return node.id, 0.0

        results = await asyncio.gather(*(score_task(node) for node in nodes_to_score))

        for node_id, score in results:
            if node_id in grs.all_nodes:
                grs.all_nodes[node_id].score = score
                grs.all_nodes[node_id].visits += 1

        logger.info(f"ScoreOp evaluated {len(nodes_to_score)} nodes in set '{target_set}'.")

    def execute(
        self,
        grs: GraphReasoningState,
        llm: BaseLLM,
        prompts: Dict[str, str],
        embedder: Optional[BaseEmbedder] = None,
        **global_kwargs: Any,
    ) -> None:
        raise NotImplementedError("Synchronous execute not fully implemented for ScoreOp sketch.")


class KeepBestOp(GoTOperation):
    """Selects the top N nodes based on score."""

    async def execute_async(
        self,
        grs: GraphReasoningState,
        llm: BaseLLM,
        prompts: Dict[str, str],
        embedder: Optional[BaseEmbedder] = None,
        semaphore: Optional[asyncio.Semaphore] = None,
        trace: Optional[Trace] = None,
        **global_kwargs: Any,
    ) -> None:
        """Selects best nodes (synchronous logic sufficient)."""
        self.execute(grs, llm, prompts, embedder, trace=trace, **global_kwargs)

    def execute(
        self,
        grs: GraphReasoningState,
        llm: BaseLLM,
        prompts: Dict[str, str],
        embedder: Optional[BaseEmbedder] = None,
        trace: Optional[Trace] = None,
        **global_kwargs: Any,
    ) -> None:
        """Selects best nodes."""
        n_best = self.params.get("N", 1)
        target_set = self.params.get("target_set", "scored")  # Operate on previously scored nodes
        output_set = self.params.get("output_set", "frontier")  # Update the main frontier

        nodes_to_consider = grs.get_active_set(target_set)
        nodes_to_consider.sort(key=lambda n: n.score, reverse=True)
        best_nodes = nodes_to_consider[:n_best]

        if trace:
            for node in best_nodes:
                trace.add_node(
                    node_id=f"keep_{node.id}",
                    parent_id=node.id,
                    content=f"Kept node {node.id} with score {node.score:.2f}",
                    score=node.score,
                    metadata={"op": "KeepBest", "N": n_best, "target_set": target_set},
                )

        grs.set_active_set(output_set, best_nodes)
        logger.info(f"KeepBestOp selected top {len(best_nodes)} nodes into set '{output_set}'.")


# --- Main GraphOfThoughts Class ---
class GraphOfThoughts:
    """Implements the Graph of Thoughts (GoT) prompting framework.

    GoT represents the reasoning process as a graph where nodes are partial solutions
    (thoughts) and edges represent dependencies or transformations between them.
    It allows for applying operations like generation, aggregation, scoring,
    and selection according to a defined Graph of Operations (GoO).

    Reference:
        Besta et al. (v4; 2024) "Graph of Thoughts: Solving Elaborate Problems with Large Language Models".
        https://arxiv.org/abs/2308.09687
    """

    def __init__(
        self,
        llm: BaseLLM,
        embedder: Optional[BaseEmbedder] = None,
        final_answer_format: Literal["text", "json", "direct_content"] = "text",
        prompts: Optional[Dict[str, str]] = None,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initializes the GraphOfThoughts strategy handler.

        Args:
            llm: The language model instance.
            embedder: Optional embedding model instance for similarity checks.
            final_answer_format: Whether to extract the final answer as raw text, JSON, or directly from the best node content.
            prompts: A dictionary mapping operation types (e.g., 'expand', 'evaluate',
                     'aggregate', 'improve') to their prompt templates.
            max_tokens: Default maximum tokens for LLM generation calls.
            seed: Default random seed for LLM calls.
        """
        self.llm = llm
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.final_answer_format = final_answer_format
        self.prompts = prompts or self._get_default_prompts()
        self.max_tokens = max_tokens
        self.seed = seed

    def _get_default_prompts(self) -> Dict[str, str]:
        """Provides default prompt templates."""
        return {
            "expand": (
                "Generate {k} distinct reasoning steps or thoughts to continue "
                "from the context below. Return ONLY a JSON object with a SINGLE KEY named 'thoughts' "
                "whose value is a list of strings.\n"
                "Context:\n{ctx}\n\nJSON Output:"
            ),
            "evaluate": (
                "Evaluate the quality of the reasoning path below on a scale of 1-10 "
                "(1=bad, 10=excellent). Return response as a JSON object with keys "
                '"score" (int) and "justification" (str).\n'
                "Path:\n{steps}\n\nJSON Evaluation:"
            ),
            "aggregate": (
                "Combine the information from the following inputs into {k} synthesized thought(s). "
                "Maximize coherence and completeness. Return as a JSON list of strings.\n"
                "Inputs:\n{context}\n\nJSON Output:"
            ),
            "improve": (
                "Improve the following thought based on the initial query and context. "
                "Return {k} improved versions as a JSON list of strings.\n"
                "Original Thought:\n{ctx}\n\nJSON Output:"
            ),
            # Add other prompts as needed
        }

    def _find_similar_node(
        self, new_node: GoTNode, nodes_to_check: List[GoTNode], threshold: float
    ) -> Optional[GoTNode]:
        """Finds an existing node similar to `new_node` based on embedding similarity.

        Args:
            new_node: The node to check for similarity.
            nodes_to_check: A list of existing nodes to compare against.
            threshold: The cosine similarity threshold for merging.

        Returns:
            The similar node if found above the threshold, otherwise None.
        """
        if not self.embedder or new_node.embed is None:
            logger.debug(
                f"Skipping similarity check for node {new_node.id} (no embedder or embedding)."
            )
            return None

        new_norm = np.linalg.norm(new_node.embed)
        if new_norm < 1e-9:
            logger.debug(f"Skipping similarity check for node {new_node.id} (zero norm embedding).")
            return None

        logger.debug(
            f"Checking similarity for node {new_node.id} against {len(nodes_to_check)} nodes."
        )
        for other in nodes_to_check:
            if other.id == new_node.id or other.embed is None:
                continue

            other_norm = np.linalg.norm(other.embed)
            if other_norm < 1e-9 or new_node.is_ancestor(other):
                continue

            try:
                embed1 = new_node.embed.ravel()
                embed2 = other.embed.ravel()
                if embed1.shape != embed2.shape:
                    logger.warning(f"Embedding shape mismatch: {embed1.shape} vs {embed2.shape}")
                    continue

                dot_product = np.dot(embed1, embed2)
                sim = float(dot_product / (new_norm * other_norm))
            except ValueError as e:
                logger.warning(
                    f"Error calculating similarity between node {new_node.id} ({embed1.shape}) and {other.id} ({embed2.shape}): {e}"
                )
                continue

            if sim > threshold:
                logger.info(
                    f"Potential merge: node {new_node.id} similar to node {other.id} (similarity: {sim:.3f})"
                )
                return other
        return None

    def _create_operation(
        self,
        op_name: str,
        params: Dict,
        custom_operations: Optional[Dict[str, "GoTOperation"]] = None,
    ) -> GoTOperation:
        """Factory method to create operation instances."""
        if custom_operations and op_name in custom_operations:
            op_class = custom_operations[op_name]
            return op_class(**params)

        if op_name == "Generate":
            return GenerateOp(**params)
        elif op_name == "Aggregate":
            return AggregateOp(**params)
        elif op_name == "Improve":
            raise NotImplementedError(f"Operation '{op_name}' not implemented yet.")
        elif op_name == "Score":
            return ScoreOp(**params)
        elif op_name == "KeepBest":
            return KeepBestOp(**params)
        # Add other operations like Validate, etc.
        else:
            raise ValueError(f"Unknown GoT operation: {op_name}")

    async def run_async(
        self,
        question: str,
        graph_of_operations: List[Tuple[str, Dict]],
        custom_operations: Optional[Dict[str, "GoTOperation"]] = None,
        semaphore: Optional[asyncio.Semaphore] = None,
        with_trace: bool = False,
        **kwargs: Any,
    ) -> Union[str, Tuple[str, Trace]]:
        """Asynchronously executes the Graph of Thoughts reasoning process based on a GoO.

        Args:
            question: The initial question or problem statement.
            graph_of_operations: A list defining the sequence of operations and their parameters.
                                 Example: [('Generate', {'k': 5, 'output_set': 'thoughts1'}),
                                           ('Score', {'target_set': 'thoughts1'}),
                                           ('KeepBest', {'N': 3, 'target_set': 'thoughts1', 'output_set': 'frontier'}),
                                           ('Aggregate', {'target_sets': ['frontier'], 'k': 1, 'output_set': 'aggregated'}),
                                           ...]
            custom_operations: An optional dictionary mapping names to custom GoTOperation classes.
            semaphore: Optional asyncio.Semaphore to limit concurrent LLM calls.
            with_trace: If True, returns a tuple of (answer, trace).
            **kwargs: Additional arguments passed to internal LLM calls (e.g., seed, max_tokens).

        Returns:
            The final answer string, or a tuple (answer, trace) if with_trace is True.
        """
        GoTNode._id_counter = 0
        root = GoTNode([question], embedder=None, text_content=question)  # Embed root optionally
        grs = GraphReasoningState(root)
        trace: Optional[Trace] = None
        if with_trace:
            trace = Trace(root_node_id=root.id)
            trace.add_node(
                node_id=root.id, parent_id=None, content=root.text_content or "", score=root.score
            )

        global_llm_params = {
            "seed": kwargs.pop("seed", self.seed),
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
            **kwargs,  # Pass remaining kwargs
        }

        logger.info(f"Starting GoT run (async) with {len(graph_of_operations)} operations.")

        for op_name, op_params in graph_of_operations:
            logger.info(f"Executing GoO Step: {op_name} with params {op_params}")
            try:
                operation = self._create_operation(op_name, op_params, custom_operations)
                await operation.execute_async(
                    grs=grs,
                    llm=self.llm,
                    prompts=self.prompts,
                    embedder=self.embedder,
                    semaphore=semaphore,
                    trace=trace,
                    **global_llm_params,
                )
            except Exception as e:
                logger.error(f"Error executing operation {op_name}: {e}", exc_info=True)
                err_msg = f"Error during operation {op_name}"
                return (err_msg, trace) if with_trace and trace else err_msg

            # Optional: Add logging for GRS state after each step
            # logger.debug(f"GRS after {op_name}: {grs.active_sets}")

        # Determine final result - assumes the relevant result is in 'frontier' or last output set
        final_candidates = grs.get_active_set(
            "frontier"
        )  # Or use a specific output set name from GoO
        if not final_candidates:
            # Fallback if the frontier is empty - check last known generated/aggregated set etc.
            # This needs robust handling based on GoO structure
            logger.warning("Frontier is empty, checking all nodes...")
            final_candidates = list(grs.all_nodes.values())

        if not final_candidates:
            logger.error("No candidate nodes found at the end of GoT run (async).")
            err_msg = "Error: No reasoning paths generated."
            return (err_msg, trace) if with_trace and trace else err_msg

        # Select the best node based on score (or other criteria if defined)
        best_node = max(final_candidates, key=lambda n: n.score)
        logger.info(f"Selected best node (async): {best_node}")

        # Use the best node's text_content for the final answer generation
        reasoning_context = best_node.text_content or "No final thought content available."
        # Or, reconstruct path if needed: numbered_reasoning = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(best_node.steps))
        final_prompt = f"Based on the final reasoning or result:\n{reasoning_context}\n\nAnswer the original question: {question}"
        logger.debug(f"Final prompt (async):\n{final_prompt}")

        try:
            local_kwargs_final = global_llm_params.copy()
            final_seed = local_kwargs_final.pop("seed", self.seed)
            final_max_tokens = local_kwargs_final.pop("max_tokens", self.max_tokens)
            answer = ""

            if self.final_answer_format == "direct_content":
                answer = best_node.text_content or "Error: Best node had no content."

            elif self.final_answer_format == "json":
                json_req = (
                    final_prompt
                    + '\n\nReturn exactly one JSON object with a single key "final_answer" whose value is the answer string.\n\nJSON Answer:'
                )
                gen_args = {
                    "response_model": ExtractedAnswer,
                    "max_tokens": final_max_tokens,
                    "seed": final_seed,
                    **local_kwargs_final,
                }
                if semaphore:
                    async with semaphore:
                        parsed = await self.llm.generate_json_async(json_req, **gen_args)
                else:
                    parsed = await self.llm.generate_json_async(json_req, **gen_args)
                # Type narrowing to ensure mypy knows we have ExtractedAnswer
                if isinstance(parsed, ExtractedAnswer):
                    final_answer_value = parsed.final_answer
                else:
                    final_answer_value = getattr(parsed, "final_answer", None)
                if isinstance(final_answer_value, str):
                    answer = final_answer_value.strip()
                elif final_answer_value is not None:
                    answer = str(final_answer_value)
                else:
                    logger.warning("GoT final async JSON extraction returned None.")
                    answer = ""
            else:
                gen_args = {
                    "max_tokens": final_max_tokens,
                    "seed": final_seed,
                    **local_kwargs_final,
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

    def run(self, question: str, graph_of_operations: List[Tuple[str, Dict]], **kwargs: Any) -> str:
        """Synchronous execution is not supported for GraphOfThoughts."""
        raise NotImplementedError(
            "Synchronous execution (run()) is not supported for GraphOfThoughts due to its "
            "reliance on internal async operations and potential event loop conflicts. "
            "Please use the asynchronous run_async() method within an async context instead."
        )
