import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .model import BaseLLM
from .schemas import EvaluationResult, ExtractedAnswer
from .utils import encode

logger = logging.getLogger(__name__)


def _strip_fences(text: str) -> str:
    """
    Removes Markdown code fences (```) from the start and end of a string.
    Handles optional language specifiers like ```json.
    """
    t = text.strip()
    # Match ``` optionally followed by 'json' then newline, capture content, end with ```
    match = re.match(r"```(?:json)?\s*(.*)\s*```", t, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Basic check for ``` at start and end
    if t.startswith("```") and t.endswith("```"):
        return t[3:-3].strip()
    # Return original text if no fences found
    return t


class GraphOfThoughts:
    """
    Implements the Graph of Thoughts (GoT) reasoning strategy.

    GoT explores potential reasoning paths as a graph, allowing for merging
    similar states and expanding promising nodes using beam search.
    """

    class _Node:
        """Internal class representing a node in the reasoning graph."""

        # Use slots for minor memory optimization on potentially many nodes
        __slots__ = ("id", "steps", "parents", "children", "embed", "visits", "score_sum", "data")
        _id_counter = 0  # Class variable for generating unique node IDs

        def __init__(
            self,
            steps: List[str],
            parents: Optional[List["GraphOfThoughts._Node"]] = None,
            data: Optional[Any] = None,
        ):
            self.id = GraphOfThoughts._Node._id_counter
            GraphOfThoughts._Node._id_counter += 1

            self.steps = steps
            self.parents = parents or []
            self.children: List["GraphOfThoughts._Node"] = []
            self.embed: Optional[np.ndarray] = None
            self.visits = 0
            self.score_sum = 0.0
            self.data = data

            try:
                text_to_encode = " -> ".join(self.steps)
                if text_to_encode:
                    emb_list = encode([text_to_encode])
                    # --- FIXED Truthiness Check ---
                    # Check if list is not empty AND first element is not None
                    if len(emb_list) > 0 and emb_list[0] is not None:
                        self.embed = np.array(emb_list[0], dtype=float)
                    # --- End Fixed Check ---
            except Exception as e:
                logger.error("Failed to encode node %d steps: %s", self.id, e)
                self.embed = None

        def score(self) -> float:
            """Calculates the average evaluation score for this node."""
            return self.score_sum / self.visits if self.visits > 0 else 0.0

        def is_ancestor(self, potential_ancestor: "GraphOfThoughts._Node") -> bool:
            """
            Checks if potential_ancestor is an ancestor of this node.
            Used to prevent merging cycles.
            """
            if not self.parents:  # Node with no parents cannot have ancestors
                return False
            queue = list(self.parents)  # Start BFS/DFS from parents
            visited = {self.id}  # Keep track of visited nodes to handle graph cycles
            while queue:
                p = queue.pop(0)  # Use pop(0) for BFS, pop() for DFS
                if p.id == potential_ancestor.id:
                    return True
                if p.id not in visited:
                    visited.add(p.id)
                    queue.extend(p.parents)  # Add parents of the current node to the queue
            return False  # Reached end without finding the potential ancestor

        def __repr__(self) -> str:
            """Provides a concise string representation of the node."""
            pids = [p.id for p in self.parents]
            return (
                f"Node(id={self.id}, steps={len(self.steps)}, "
                f"score={self.score():.2f}, visits={self.visits}, parents={pids})"
            )

    def __init__(
        self,
        llm: BaseLLM,
        max_iters: int = 5,
        num_branches: int = 5,
        beam_width: int = 3,
        merge_threshold: float = 0.9,
        expand_prompt: str = (
            "Generate {k} distinct reasoning steps or thoughts to continue "
            "from the context below. Return as a JSON list of strings.\n"
            "Context:\n{ctx}\n\nJSON Steps:"
        ),
        eval_prompt: str = (
            "Evaluate the quality of the reasoning path below on a scale of 1-10 "
            "(1=bad, 10=excellent). Return response as a JSON object with keys "
            '"score" (int) and "justification" (str).\n'
            "Path:\n{steps}\n\nJSON Evaluation:"
        ),
        # Note: 'use_json' controls *both* expansion format and final answer format.
        # Consider splitting if more granular control is needed.
        use_json: bool = False,  # If True, expects JSON list for expansion and produces JSON answer
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """
        Initializes the GraphOfThoughts instance.

        Args:
            llm: The language model instance.
            max_iters: Maximum number of expansion iterations.
            num_branches: Number of new thoughts/steps to generate at each expansion.
            beam_width: Number of best nodes to keep in the frontier at each iteration.
            merge_threshold: Cosine similarity threshold for merging similar nodes.
            expand_prompt: Prompt template for generating new reasoning steps. Needs {k} and {ctx}.
            eval_prompt: Prompt template for evaluating a reasoning path. Needs {steps}.
            use_json: If True, assumes expansion results are JSON lists and generates final answer as JSON.
                      If False, parses expansion as JSON (see _parse) but generates final answer as text.
            max_tokens: Optional max tokens for LLM generation calls.
            seed: Optional random seed for LLM generation.
        """
        self.llm = llm
        # --- REMOVED _raw_llm ---
        # No longer attempt to bypass wrappers here. Assume self.llm works as expected.
        # self._raw_llm = getattr(llm, "_real", llm)
        # --- END REMOVED ---

        self.max_iters = max_iters
        self.num_branches = num_branches
        self.beam_width = beam_width
        self.merge_threshold = merge_threshold
        self.expand_prompt = expand_prompt
        self.eval_prompt = eval_prompt
        self.use_json = use_json  # Controls expansion parsing and final answer format

        self.max_tokens = max_tokens
        self.seed = seed

    def _parse(self, raw: str) -> List[str]:
        """
        Parses the raw LLM output expected during the expansion phase.
        Tries to interpret the output as a JSON list of strings,
        or a JSON object containing a 'thoughts' key with a list of strings.

        Args:
            raw: The raw string output from the LLM during expansion.

        Returns:
            A list of extracted thought strings, limited by `self.num_branches`.
            Returns an empty list if parsing fails.
        """
        raw_stripped = _strip_fences(raw)  # Remove potential markdown fences
        try:
            parsed_obj = json.loads(raw_stripped)  # Attempt to parse as JSON
            thought_list: Optional[List[Any]] = None

            # Check if it's a dict with a 'thoughts' key
            if isinstance(parsed_obj, dict) and "thoughts" in parsed_obj:
                if isinstance(parsed_obj["thoughts"], list):
                    thought_list = parsed_obj["thoughts"]
            # Check if it's directly a list
            elif isinstance(parsed_obj, list):
                thought_list = parsed_obj
            else:
                # Parsed JSON is not in expected format
                logger.warning(
                    f"Parsed JSON is not a list or dict with 'thoughts': {type(parsed_obj)}"
                )
                return []

            # Filter and clean up the extracted thoughts
            if thought_list is not None:
                valid_thoughts = [
                    str(s).strip()  # Convert to string and strip whitespace
                    for s in thought_list
                    # Check type and ensure not empty after stripping
                    if isinstance(s, (str, int, float)) and str(s).strip()
                ]
                # Limit the number of thoughts returned
                return valid_thoughts[: self.num_branches]
            else:
                return []

        except json.JSONDecodeError as e:
            logger.error(
                "Failed to parse expansion JSON: %s\nRaw Stripped: %s", e, raw_stripped[:200]
            )
            return []  # Return empty list on parsing failure
        except Exception as e:  # Catch other potential errors
            logger.error("Unexpected error during expansion parsing: %s", e, exc_info=True)
            return []

    def _evaluate(self, steps: List[str]) -> float:
        """
        Evaluates a given path (list of steps) using the LLM synchronously.

        Args:
            steps: The list of reasoning steps to evaluate.

        Returns:
            A score between 0.0 and 1.0, based on the LLM evaluation.
            Returns 0.0 if evaluation fails.
        """
        if not steps:  # Cannot evaluate empty steps
            return 0.0
        # Format steps for the prompt
        numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
        prompt = self.eval_prompt.format(steps=numbered)
        logger.debug(f"Evaluating node steps (sync) with prompt:\n{prompt}")

        try:
            # Use generate_json to get a structured response
            result = self.llm.generate_json(
                prompt,
                response_model=EvaluationResult,  # Expect EvaluationResult schema
                max_tokens=self.max_tokens,
                seed=self.seed,
            )
            # Ensure result is of the expected type (generate_json should guarantee this on success)
            if isinstance(result, EvaluationResult):
                score = float(result.score)
                # Normalize score from 1-10 range to 0-1 range
                normalized_score = max(0.0, min(1.0, (score - 1.0) / 9.0))
                logger.debug(
                    f"Evaluation score: {score} -> Normalized: {normalized_score:.3f}. Justification: {result.justification}"
                )
                return normalized_score
            else:
                # Should not happen if generate_json works correctly
                logger.error(f"Evaluation returned unexpected type: {type(result)}")
                return 0.0
        except Exception as e:
            logger.error("Evaluation LLM call failed: %s", e, exc_info=True)
            return 0.0  # Return default score on error

    async def _evaluate_async(
        self, steps: List[str], semaphore: Optional[asyncio.Semaphore] = None
    ) -> float:
        """
        Evaluates a given path (list of steps) using the LLM asynchronously.

        Args:
            steps: The list of reasoning steps to evaluate.
            semaphore: Optional semaphore to limit concurrent LLM calls.

        Returns:
            A score between 0.0 and 1.0, based on the LLM evaluation.
            Returns 0.0 if evaluation fails.
        """
        if not steps:
            return 0.0
        # Format steps for the prompt
        numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
        prompt = self.eval_prompt.format(steps=numbered)
        logger.debug(f"Evaluating node steps (async) with prompt:\n{prompt}")

        try:
            # Use generate_json_async with semaphore if provided
            if semaphore:
                async with semaphore:
                    result = await self.llm.generate_json_async(
                        prompt,
                        response_model=EvaluationResult,
                        max_tokens=self.max_tokens,
                        seed=self.seed,
                    )
            else:
                result = await self.llm.generate_json_async(
                    prompt,
                    response_model=EvaluationResult,
                    max_tokens=self.max_tokens,
                    seed=self.seed,
                )

            # Process result (same logic as sync version)
            if isinstance(result, EvaluationResult):
                score = float(result.score)
                normalized_score = max(0.0, min(1.0, (score - 1.0) / 9.0))
                logger.debug(
                    f"Async evaluation score: {score} -> Normalized: {normalized_score:.3f}. Justification: {result.justification}"
                )
                return normalized_score
            else:
                logger.error(f"Async evaluation returned unexpected type: {type(result)}")
                return 0.0
        except Exception as e:
            logger.error("Async evaluation LLM call failed: %s", e, exc_info=True)
            return 0.0

    def _find_similar_node(self, new_node: _Node, nodes_to_check: List[_Node]) -> Optional[_Node]:
        """
        Finds an existing node in `nodes_to_check` that is similar to `new_node`.

        Similarity is based on cosine similarity of node embeddings exceeding `merge_threshold`.
        Prevents merging with ancestors.

        Args:
            new_node: The newly created node.
            nodes_to_check: List of existing nodes to compare against.

        Returns:
            The similar existing node if found, otherwise None.
        """
        if new_node.embed is None:  # Cannot compare if new node has no embedding
            logger.debug(f"Skipping similarity check for node {new_node.id} (no embedding).")
            return None

        new_norm = np.linalg.norm(new_node.embed)
        if new_norm == 0:  # Cannot compare if embedding norm is zero
            logger.debug(f"Skipping similarity check for node {new_node.id} (zero norm embedding).")
            return None

        logger.debug(
            f"Checking similarity for node {new_node.id} against {len(nodes_to_check)} nodes."
        )
        for other in nodes_to_check:
            # Skip comparison with self or nodes without embeddings
            if other.id == new_node.id or other.embed is None:
                continue

            other_norm = np.linalg.norm(other.embed)
            # Skip if other node has zero norm or if merging would create a cycle
            if other_norm == 0 or new_node.is_ancestor(other):
                continue

            # Calculate cosine similarity
            try:
                dot_product = np.dot(new_node.embed.flatten(), other.embed.flatten())
                sim = float(dot_product / (new_norm * other_norm))
            except ValueError as e:
                logger.warning(
                    f"Error calculating similarity between node {new_node.id} and {other.id}: {e}"
                )
                continue  # Skip comparison if dimensions mismatch

            # Check if similarity exceeds threshold
            if sim > self.merge_threshold:
                logger.info(
                    f"Merging node {new_node.id} into similar node {other.id} (similarity: {sim:.3f})"
                )
                return other  # Found a similar node

        return None  # No similar node found

    def run(self, question: str) -> str:
        """
        Runs the Graph of Thoughts algorithm synchronously.

        Args:
            question: The initial question or problem statement.

        Returns:
            The final generated answer string.
        """
        # Reset node counter for this run
        GraphOfThoughts._Node._id_counter = 0
        root = self._Node([question])  # Start with the question as the root node
        frontier = [root]  # Nodes to expand in the current iteration (beam)
        all_nodes = {root.id: root}  # Dictionary to store all created nodes by ID

        logger.info(
            f"Starting GoT run. Max iterations: {self.max_iters}, Beam width: {self.beam_width}"
        )

        for iter_num in range(self.max_iters):
            logger.info(f"--- GoT Iteration {iter_num + 1}/{self.max_iters} ---")
            logger.debug(f"Frontier size: {len(frontier)}. Nodes: {[n.id for n in frontier]}")
            if not frontier:
                logger.info("Frontier is empty. Stopping iterations.")
                break

            expansion_results: Dict[
                int, List[str]
            ] = {}  # Store expansion results {node_id: [thoughts]}

            # --- Expansion Phase ---
            logger.info(f"Expanding {len(frontier)} nodes in the frontier...")
            for node in frontier:
                # Format context from node's steps
                ctx = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(node.steps))
                prompt = self.expand_prompt.format(k=self.num_branches, ctx=ctx)
                exps: List[str] = []  # Initialize list for expansion thoughts

                try:
                    # --- USE self.llm DIRECTLY ---
                    # Decide based on use_json flag how to call and parse
                    if self.use_json:
                        # If use_json is True, assume we want a JSON list back
                        # We need to prompt for JSON specifically
                        json_p = (
                            prompt + "\n\nReturn exactly one JSON list of strings.\n\nJSON Answer:"
                        )
                        logger.debug(f"Expanding node {node.id} using JSON prompt.")
                        # We could use generate_json here, but _parse handles list/dict ambiguity
                        raw = self.llm.generate(
                            json_p,
                            max_tokens=self.max_tokens,
                            seed=(self.seed + iter_num + node.id)
                            if self.seed is not None
                            else None,
                        )
                        exps = self._parse(raw)  # Parse the expected JSON list/dict
                    else:
                        # If use_json is False, generate raw text and attempt to parse it as if it were JSON
                        # This relies on the LLM following instructions from `expand_prompt` implicitly
                        logger.debug(
                            f"Expanding node {node.id} using standard prompt (expecting parsable output)."
                        )
                        raw = self.llm.generate(
                            prompt,
                            max_tokens=self.max_tokens,
                            seed=(self.seed + iter_num + node.id)
                            if self.seed is not None
                            else None,
                        )
                        exps = self._parse(raw)  # _parse attempts to find JSON list/dict
                    # --- END self.llm USAGE ---
                    logger.debug(f"Node {node.id} expanded into {len(exps)} thoughts.")
                except Exception as e:
                    logger.error(f"Expansion failed for node {node.id}: {e}", exc_info=True)
                    exps = []  # Ensure exps is empty on error

                expansion_results[node.id] = exps

            # --- Node Creation, Merging, and Evaluation Phase ---
            newly_added: List[
                GraphOfThoughts._Node
            ] = []  # Keep track of genuinely new nodes this iteration
            for node in frontier:  # Iterate through the nodes that were expanded
                for step in expansion_results.get(node.id, []):  # For each generated thought
                    # Create a potential new node
                    new_node = self._Node(node.steps + [step], parents=[node])

                    # Check for similarity with existing nodes to potentially merge
                    similar = self._find_similar_node(new_node, list(all_nodes.values()))

                    if similar:
                        # Merge: Add current node as another parent to the similar existing node
                        if node not in similar.parents:
                            similar.parents.append(node)
                            logger.debug(
                                f"Added node {node.id} as parent to existing node {similar.id}"
                            )
                        # Do not add the new_node to the graph or newly_added list
                        # (Decrement counter if IDs are strictly sequential and we skip one?)
                        # GraphOfThoughts._Node._id_counter -= 1 # Optional: Reclaim ID if desired
                        continue  # Skip to the next step/thought
                    else:
                        # No similar node found: Add the new node to the graph
                        node.children.append(new_node)
                        all_nodes[new_node.id] = new_node
                        newly_added.append(new_node)
                        logger.debug(f"Added new node {new_node.id} from parent {node.id}")

            if not newly_added:
                logger.info("No new nodes were added in this iteration. Stopping.")
                break  # Stop if no progress is made

            # Evaluate the newly added nodes
            logger.info(f"Evaluating {len(newly_added)} newly added nodes...")
            scored_nodes: List[Tuple[float, GraphOfThoughts._Node]] = []
            for n in newly_added:
                # Evaluate the path leading to the new node
                node_score = self._evaluate(n.steps)
                # Update node's score statistics
                n.visits += 1
                n.score_sum += node_score
                scored_nodes.append((n.score(), n))  # Store (score, node) tuple

            # --- Pruning / Beam Search Phase ---
            # Sort nodes by score (descending)
            scored_nodes.sort(key=lambda x: x[0], reverse=True)
            # Select the top `beam_width` nodes for the next frontier
            frontier = [node for score, node in scored_nodes[: self.beam_width]]
            logger.info(
                f"Selected top {len(frontier)} nodes for next frontier (Beam Width: {self.beam_width})."
            )

            if not frontier:
                logger.info("Frontier became empty after pruning. Stopping.")
                break  # Stop if beam becomes empty

        # --- Final Answer Generation ---
        # Select the best node overall
        # If frontier is empty after loop, consider all nodes, otherwise just the final frontier
        final_candidates = frontier or list(all_nodes.values())
        if not final_candidates:
            logger.error("No candidate nodes found at the end of GoT run.")
            return "Error: No reasoning paths generated."

        # Find the node with the highest score among candidates
        best_node = max(final_candidates, key=lambda n: n.score())
        logger.info(f"Selected best node: {best_node}")

        # Prepare the final prompt using the reasoning steps from the best node
        # Exclude the initial question from the reasoning steps shown
        reasoning = (
            best_node.steps[1:]
            if len(best_node.steps) > 1
            else ["No intermediate steps generated."]
        )
        numbered_reasoning = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(reasoning))
        final_prompt = (
            f"Given reasoning steps:\n{numbered_reasoning}\n\nAnswer the question: {question}"
        )
        logger.debug(f"Final prompt:\n{final_prompt}")

        # Generate the final answer using the LLM
        try:
            if self.use_json:
                # If use_json is True, request the final answer as JSON
                json_req = (
                    final_prompt
                    + '\n\nReturn exactly one JSON object with a single key "final_answer" whose value is the answer string.\n\nJSON Answer:'
                )
                parsed = self.llm.generate_json(
                    json_req,
                    response_model=ExtractedAnswer,
                    max_tokens=self.max_tokens,
                    seed=self.seed,  # Use seed for final generation too
                )
                return parsed.final_answer.strip()
            else:
                # If use_json is False, generate a standard text answer
                return self.llm.generate(
                    final_prompt,
                    max_tokens=self.max_tokens,
                    seed=self.seed,
                ).strip()
        except Exception as e:
            logger.error("Final answer generation failed: %s", e, exc_info=True)
            return "Error generating final answer."

    async def run_async(self, question: str, semaphore: Optional[asyncio.Semaphore] = None) -> str:
        """
        Runs the Graph of Thoughts algorithm asynchronously.

        Args:
            question: The initial question or problem statement.
            semaphore: Optional semaphore to limit concurrent LLM calls.

        Returns:
            The final generated answer string.
        """
        # Reset node counter for this run
        GraphOfThoughts._Node._id_counter = 0
        root = self._Node([question])
        frontier = [root]
        all_nodes = {root.id: root}

        logger.info(
            f"Starting GoT run (async). Max iterations: {self.max_iters}, Beam width: {self.beam_width}"
        )

        for iter_num in range(self.max_iters):
            logger.info(f"--- GoT Iteration {iter_num + 1}/{self.max_iters} (async) ---")
            logger.debug(f"Frontier size: {len(frontier)}. Nodes: {[n.id for n in frontier]}")
            if not frontier:
                logger.info("Frontier is empty. Stopping iterations.")
                break

            # --- Async Expansion Phase ---
            logger.info(f"Expanding {len(frontier)} nodes in the frontier asynchronously...")

            async def expand_task(node: GraphOfThoughts._Node) -> Tuple[int, List[str]]:
                # Format context
                ctx = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(node.steps))
                prompt = self.expand_prompt.format(k=self.num_branches, ctx=ctx)
                exps: List[str] = []

                try:
                    # --- USE self.llm DIRECTLY (async) ---
                    gen_seed = (self.seed + iter_num + node.id) if self.seed is not None else None
                    if self.use_json:
                        json_p = (
                            prompt + "\n\nReturn exactly one JSON list of strings.\n\nJSON Answer:"
                        )
                        logger.debug(f"Expanding node {node.id} using JSON prompt (async).")
                        if semaphore:
                            async with semaphore:
                                raw = await self.llm.generate_async(
                                    json_p, max_tokens=self.max_tokens, seed=gen_seed
                                )
                        else:
                            raw = await self.llm.generate_async(
                                json_p, max_tokens=self.max_tokens, seed=gen_seed
                            )
                        exps = self._parse(raw)
                    else:
                        logger.debug(f"Expanding node {node.id} using standard prompt (async).")
                        if semaphore:
                            async with semaphore:
                                raw = await self.llm.generate_async(
                                    prompt, max_tokens=self.max_tokens, seed=gen_seed
                                )
                        else:
                            raw = await self.llm.generate_async(
                                prompt, max_tokens=self.max_tokens, seed=gen_seed
                            )
                        exps = self._parse(raw)
                    # --- END self.llm USAGE ---
                    logger.debug(f"Node {node.id} expanded into {len(exps)} thoughts (async).")
                except Exception as e:
                    logger.error(f"Async expansion failed for node {node.id}: {e}", exc_info=True)
                    exps = []

                return node.id, exps

            # Run expansion tasks concurrently
            results = await asyncio.gather(
                *(expand_task(n) for n in frontier), return_exceptions=True
            )
            expansion_results: Dict[int, List[str]] = {}
            for i, res in enumerate(results):
                node_id = frontier[i].id
                if isinstance(res, Exception):
                    logger.error(f"Async expansion task failed for node {node_id}: {res}")
                    expansion_results[node_id] = []
                elif isinstance(res, tuple) and len(res) == 2:
                    expansion_results[res[0]] = res[1]  # Store {node_id: [thoughts]}
                else:
                    logger.error(
                        f"Unexpected result type from expand_task for node {node_id}: {type(res)}"
                    )
                    expansion_results[node_id] = []

            # --- Node Creation, Merging Phase (Sync - typically fast) ---
            newly_added: List[GraphOfThoughts._Node] = []
            for nid, steps in expansion_results.items():
                # Check if parent node exists (it should, unless error occurred)
                parent = all_nodes.get(nid)
                if parent is None:
                    logger.warning(f"Parent node {nid} not found, cannot add children.")
                    continue

                for step in steps:
                    new_node = self._Node(parent.steps + [step], parents=[parent])
                    similar = self._find_similar_node(new_node, list(all_nodes.values()))
                    if similar:
                        if parent not in similar.parents:
                            similar.parents.append(parent)
                        continue
                    parent.children.append(new_node)
                    all_nodes[new_node.id] = new_node
                    newly_added.append(new_node)

            if not newly_added:
                logger.info("No new nodes were added in this iteration (async). Stopping.")
                break

            # --- Async Evaluation Phase ---
            logger.info(f"Evaluating {len(newly_added)} newly added nodes asynchronously...")

            async def eval_task(node: GraphOfThoughts._Node) -> Tuple[int, float]:
                # Evaluate the path leading to this node
                node_score = await self._evaluate_async(node.steps, semaphore)
                return node.id, node_score

            # Run evaluation tasks concurrently
            score_results = await asyncio.gather(
                *(eval_task(n) for n in newly_added), return_exceptions=True
            )

            # Process evaluation results and update nodes
            scored_nodes: List[Tuple[float, GraphOfThoughts._Node]] = []
            processed_ids = set()  # Track nodes evaluated in this iteration
            for i, res in enumerate(score_results):
                node_id = newly_added[i].id
                processed_ids.add(node_id)
                node = all_nodes.get(node_id)
                if node is None:  # Should not happen
                    continue

                if isinstance(res, Exception):
                    logger.error(f"Async evaluation task failed for node {node_id}: {res}")
                    node_score = 0.0  # Assign default score on error
                elif isinstance(res, tuple) and len(res) == 2:
                    node_score = res[1]
                else:
                    logger.error(
                        f"Unexpected result type from eval_task for node {node_id}: {type(res)}"
                    )
                    node_score = 0.0

                # Update node stats
                node.visits += 1
                node.score_sum += node_score
                scored_nodes.append((node.score(), node))

            # Include nodes from previous frontier that weren't re-evaluated (though typically all new are evaluated)
            # This ensures nodes stick around if evaluation fails for all new nodes
            # for f_node in frontier:
            #     if f_node.id not in processed_ids:
            #         scored_nodes.append((f_node.score(), f_node))

            # --- Pruning / Beam Search Phase (Sync - fast) ---
            scored_nodes.sort(key=lambda x: x[0], reverse=True)
            frontier = [node for score, node in scored_nodes[: self.beam_width]]
            logger.info(
                f"Selected top {len(frontier)} nodes for next frontier (Async Beam Width: {self.beam_width})."
            )

            if not frontier:
                logger.info("Frontier became empty after pruning (async). Stopping.")
                break

        # --- Final Answer Generation (Async) ---
        final_candidates = frontier or list(all_nodes.values())
        if not final_candidates:
            logger.error("No candidate nodes found at the end of GoT run (async).")
            return "Error: No reasoning paths generated."

        best_node = max(final_candidates, key=lambda n: n.score())
        logger.info(f"Selected best node (async): {best_node}")

        reasoning = (
            best_node.steps[1:]
            if len(best_node.steps) > 1
            else ["No intermediate steps generated."]
        )
        numbered_reasoning = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(reasoning))
        final_prompt = (
            f"Given reasoning steps:\n{numbered_reasoning}\n\nAnswer the question: {question}"
        )
        logger.debug(f"Final prompt (async):\n{final_prompt}")

        try:
            final_seed = self.seed  # Use base seed for final answer
            if self.use_json:
                json_req = (
                    final_prompt
                    + '\n\nReturn exactly one JSON object with a single key "final_answer" whose value is the answer string.\n\nJSON Answer:'
                )
                # Use semaphore for the final call too, if provided
                if semaphore:
                    async with semaphore:
                        parsed = await self.llm.generate_json_async(
                            json_req,
                            response_model=ExtractedAnswer,
                            max_tokens=self.max_tokens,
                            seed=final_seed,
                        )
                else:
                    parsed = await self.llm.generate_json_async(
                        json_req,
                        response_model=ExtractedAnswer,
                        max_tokens=self.max_tokens,
                        seed=final_seed,
                    )
                return parsed.final_answer.strip()
            else:
                if semaphore:
                    async with semaphore:
                        return (
                            await self.llm.generate_async(
                                final_prompt, max_tokens=self.max_tokens, seed=final_seed
                            )
                        ).strip()
                else:
                    return (
                        await self.llm.generate_async(
                            final_prompt, max_tokens=self.max_tokens, seed=final_seed
                        )
                    ).strip()
        except Exception as e:
            logger.error("Final async answer generation failed: %s", e, exc_info=True)
            return "Error generating final async answer."

    # Allow calling the instance like a function (defaults to synchronous run)
    __call__ = run
