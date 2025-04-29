import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .model import BaseLLM
from .utils import encode

logger = logging.getLogger(__name__)


def _strip_fences(text: str) -> str:
    t = text.strip()
    # More robust regex to handle optional json language tag and varying whitespace
    match = re.match(r"```(?:json)?\s*(.*)\s*```", t, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Handle potential simple ``` fences without language tag
    if t.startswith("```") and t.endswith("```"):
        return t[3:-3].strip()
    return t  # Return original if no fences found


class GraphOfThoughts:
    class _Node:
        __slots__ = ("id", "steps", "parents", "children", "embed", "visits", "score_sum", "data")
        _id_counter = 0

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

            try:
                # Use a consistent separator for encoding
                text_to_encode = " -> ".join(self.steps)
                emb = encode([text_to_encode])[0]
                self.embed = np.array(emb, dtype=float)
            except Exception as e:
                logger.error("Failed to encode node %d steps: %s", self.id, e)
                self.embed = None

            self.visits = 0
            self.score_sum = 0.0
            self.data = data

        def score(self) -> float:
            return self.score_sum / self.visits if self.visits > 0 else 0.0

        def is_ancestor(self, potential_ancestor: "GraphOfThoughts._Node") -> bool:
            """Check if potential_ancestor is an ancestor of this node."""
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
    ):
        self.llm = llm
        self.max_iters = max_iters
        self.num_branches = num_branches
        self.beam_width = beam_width
        self.merge_threshold = merge_threshold
        self.expand_prompt = expand_prompt
        self.eval_prompt = eval_prompt

    def _parse(self, raw: str) -> List[str]:
        raw_stripped = _strip_fences(raw)
        try:
            arr = json.loads(raw_stripped)
            if not isinstance(arr, list):
                logger.warning("Parsed expansion is not a list: %r", arr)
                return []
            # Ensure elements are strings before stripping
            return [
                str(s).strip() for s in arr if isinstance(s, (str, int, float)) and str(s).strip()
            ][: self.num_branches]
        except json.JSONDecodeError as e:
            logger.error(
                "Failed to parse expansion JSON: %s\nStripped Text was: %s", e, raw_stripped[:200]
            )
            return []
        except Exception as e:  # Catch other potential errors during parsing/list comprehension
            logger.error(
                "Unexpected error parsing expansion: %s\nStripped Text was: %s",
                e,
                raw_stripped[:200],
            )
            return []

    def _evaluate(self, steps: List[str]) -> float:
        numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
        prompt = self.eval_prompt.format(steps=numbered)
        try:
            result = self.llm.generate_json(prompt)
            if not isinstance(result, dict):
                logger.warning("Evaluation result is not a dictionary: %s", result)
                return 0.0
            score = float(result.get("score", 0))
            return max(0.0, min(1.0, (score - 1.0) / 9.0))
        except (ValueError, TypeError) as e:
            logger.error(
                "Error converting evaluation score to float: %s. Result was: %s", e, result
            )
            return 0.0
        except Exception as e:
            logger.error("Evaluation error: %s", e)
            return 0.0

    async def _evaluate_async(
        self, steps: List[str], semaphore: Optional[asyncio.Semaphore] = None
    ) -> float:
        numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
        prompt = self.eval_prompt.format(steps=numbered)
        try:
            if semaphore:
                async with semaphore:
                    result = await self.llm.generate_json_async(prompt)
            else:
                result = await self.llm.generate_json_async(prompt)

            if not isinstance(result, dict):
                logger.warning("Async evaluation result is not a dictionary: %s", result)
                return 0.0
            score = float(result.get("score", 0))
            return max(0.0, min(1.0, (score - 1.0) / 9.0))
        except (ValueError, TypeError) as e:
            logger.error(
                "Error converting async evaluation score to float: %s. Result was: %s", e, result
            )
            return 0.0
        except Exception as e:
            logger.error("Async evaluation error: %s", e)
            return 0.0

    def _find_similar_node(self, new_node: _Node, nodes_to_check: List[_Node]) -> Optional[_Node]:
        if new_node.embed is None:
            return None
        new_norm = np.linalg.norm(new_node.embed)
        if new_norm == 0:
            return None

        for other in nodes_to_check:
            if other.id == new_node.id or other.embed is None:
                continue

            other_norm = np.linalg.norm(other.embed)
            if other_norm == 0:
                continue

            # Avoid merging with own ancestors
            if new_node.is_ancestor(other):
                continue

            sim = float(np.dot(new_node.embed, other.embed) / (new_norm * other_norm))

            if sim > self.merge_threshold:
                return other
        return None

    def run(self, question: str) -> str:
        GraphOfThoughts._Node._id_counter = 0
        root = self._Node([question])
        frontier: List[GraphOfThoughts._Node] = [root]
        all_nodes: Dict[int, GraphOfThoughts._Node] = {root.id: root}

        for it in range(self.max_iters):
            logger.debug("GoT iter %d/%d, frontier=%d", it + 1, self.max_iters, len(frontier))
            candidates: List[GraphOfThoughts._Node] = []
            expansion_results: Dict[int, List[str]] = {}

            # --- Expansion Phase ---
            for node in frontier:
                ctx = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(node.steps))
                prompt = self.expand_prompt.format(k=self.num_branches, ctx=ctx)
                try:
                    raw = self.llm.generate(prompt)
                    exps = self._parse(raw)
                    expansion_results[node.id] = exps
                except Exception as e:
                    logger.error("Expansion failed for node %d: %s", node.id, e)
                    expansion_results[node.id] = []  # Ensure entry exists

            # --- Candidate Generation & Merging ---
            newly_added_nodes: List[GraphOfThoughts._Node] = []
            for node in frontier:
                parent_node = all_nodes[node.id]  # Get node from central dict
                exps = expansion_results.get(parent_node.id, [])
                for step in exps:
                    new_steps = parent_node.steps + [step]
                    new_node = self._Node(new_steps, parents=[parent_node])

                    # Check for similarity *before* adding to graph structures
                    # Compare against all nodes created so far for potential merging
                    similar_node = self._find_similar_node(new_node, list(all_nodes.values()))

                    if similar_node:
                        logger.debug(
                            "Discarding node %d (similar to %d)", new_node.id, similar_node.id
                        )
                        # Optional: Link parent to the existing similar node if not already linked?
                        if new_node not in similar_node.children:  # Avoid duplicate links
                            parent_node.children.append(similar_node)  # Link parent to existing
                        # Make sure the similar node knows about this parent
                        if parent_node not in similar_node.parents:
                            similar_node.parents.append(parent_node)
                        # Do not add the new_node to candidates or all_nodes
                        continue  # Skip to next step/expansion

                    # If not merged/discarded, add to graph
                    parent_node.children.append(new_node)
                    all_nodes[new_node.id] = new_node
                    newly_added_nodes.append(new_node)

            candidates.extend(newly_added_nodes)

            if not candidates:
                logger.debug("No new candidates generated or all were merged; stopping early.")
                break

            # --- Evaluation & Pruning ---
            scored_candidates: List[Tuple[float, GraphOfThoughts._Node]] = []
            for n in candidates:
                # Ensure we evaluate the node from the central dictionary
                node_to_eval = all_nodes[n.id]
                s = self._evaluate(node_to_eval.steps)
                node_to_eval.visits += 1
                node_to_eval.score_sum += s
                scored_candidates.append((node_to_eval.score(), node_to_eval))

            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            frontier = [n for _, n in scored_candidates[: self.beam_width]]

            if not frontier:
                logger.debug("Frontier emptied by pruning; stopping.")
                break

        # --- Final Answer Generation ---
        final_candidates = frontier or list(all_nodes.values())
        if not final_candidates:
            logger.error("No nodes available to generate final answer.")
            return "Error: No reasoning paths found."

        best_node = max(final_candidates, key=lambda n: n.score())
        reasoning = best_node.steps[1:]
        numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(reasoning))
        final_prompt = f"Given reasoning steps:\n{numbered}\n\nAnswer the question: {question}"

        try:
            return self.llm.generate(final_prompt).strip()
        except Exception as e:
            logger.error("Final answer generation failed: %s", e)
            return "Error generating final answer."

    async def run_async(self, question: str, semaphore: Optional[asyncio.Semaphore] = None) -> str:
        GraphOfThoughts._Node._id_counter = 0
        root = self._Node([question])
        frontier: List[GraphOfThoughts._Node] = [root]
        all_nodes: Dict[int, GraphOfThoughts._Node] = {root.id: root}

        for it in range(self.max_iters):
            logger.debug("GoT async iter %d/%d, frontier=%d", it + 1, self.max_iters, len(frontier))
            expansion_tasks = []
            node_map: Dict[int, GraphOfThoughts._Node] = {
                n.id: n for n in frontier
            }  # Map task ID back to node

            # --- Async Expansion Phase ---
            async def expand_node_task(node_id: int):
                node = node_map[node_id]
                ctx = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(node.steps))
                prompt = self.expand_prompt.format(k=self.num_branches, ctx=ctx)
                try:
                    if semaphore:
                        async with semaphore:
                            raw = await self.llm.generate_async(prompt)
                    else:
                        raw = await self.llm.generate_async(prompt)
                    return node_id, self._parse(raw)
                except Exception as e:
                    logger.error("Async expansion failed for node %d: %s", node_id, e)
                    return node_id, []

            for node in frontier:
                expansion_tasks.append(expand_node_task(node.id))

            expansion_results_list = await asyncio.gather(*expansion_tasks)
            expansion_results: Dict[int, List[str]] = dict(expansion_results_list)

            # --- Candidate Generation & Merging ---
            newly_added_nodes: List[GraphOfThoughts._Node] = []
            for node_id, exps in expansion_results.items():
                parent_node = all_nodes[node_id]  # Get node from central dict
                for step in exps:
                    new_steps = parent_node.steps + [step]
                    new_node = self._Node(new_steps, parents=[parent_node])

                    similar_node = self._find_similar_node(new_node, list(all_nodes.values()))
                    if similar_node:
                        logger.debug(
                            "Async discard node %d (similar to %d)", new_node.id, similar_node.id
                        )
                        if new_node not in similar_node.children:
                            parent_node.children.append(similar_node)
                        if parent_node not in similar_node.parents:
                            similar_node.parents.append(parent_node)
                        continue

                    parent_node.children.append(new_node)
                    all_nodes[new_node.id] = new_node
                    newly_added_nodes.append(new_node)

            if not newly_added_nodes:
                logger.debug(
                    "No new async candidates generated or all were merged; stopping early."
                )
                break

            # --- Async Evaluation & Pruning ---
            eval_tasks = []
            candidate_map: Dict[int, GraphOfThoughts._Node] = {n.id: n for n in newly_added_nodes}

            async def evaluate_node_task(node_id: int):
                node = candidate_map[node_id]
                score = await self._evaluate_async(node.steps, semaphore)
                return node_id, score

            for node in newly_added_nodes:
                eval_tasks.append(evaluate_node_task(node.id))

            eval_results = await asyncio.gather(*eval_tasks)

            scored_candidates: List[Tuple[float, GraphOfThoughts._Node]] = []
            for node_id, score in eval_results:
                node_to_update = all_nodes[node_id]  # Update the node in the central dict
                node_to_update.visits += 1
                node_to_update.score_sum += score
                scored_candidates.append((node_to_update.score(), node_to_update))

            # Add previously existing frontier nodes that weren't re-evaluated this round
            # Their scores remain unchanged but they compete in pruning
            processed_ids = {nid for nid, _ in eval_results}
            for node in frontier:
                if node.id not in processed_ids:
                    # Ensure we use the potentially updated node from all_nodes
                    current_node_state = all_nodes[node.id]
                    scored_candidates.append((current_node_state.score(), current_node_state))

            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            frontier = [n for _, n in scored_candidates[: self.beam_width]]

            if not frontier:
                logger.debug("Async frontier emptied by pruning; stopping.")
                break

        # --- Final Answer Generation ---
        final_candidates = frontier or list(all_nodes.values())
        if not final_candidates:
            logger.error("No async nodes available to generate final answer.")
            return "Error: No reasoning paths found."

        best_node = max(final_candidates, key=lambda n: n.score())
        reasoning = best_node.steps[1:]
        numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(reasoning))
        final_prompt = f"Given reasoning steps:\n{numbered}\n\nAnswer the question: {question}"

        try:
            if semaphore:
                async with semaphore:
                    return (await self.llm.generate_async(final_prompt)).strip()
            else:
                return (await self.llm.generate_async(final_prompt)).strip()
        except Exception as e:
            logger.error("Final async answer generation failed: %s", e)
            return "Error generating final async answer."

    __call__ = run
