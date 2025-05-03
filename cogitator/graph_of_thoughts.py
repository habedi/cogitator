import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from .model import BaseLLM
from .schemas import EvaluationResult, ExtractedAnswer
from .utils import encode

logger = logging.getLogger(__name__)


def _strip_fences(text: str) -> str:
    t = text.strip()
    match = re.match(r"```(?:json)?\s*(.*)\s*```", t, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    if t.startswith("```") and t.endswith("```"):
        return t[3:-3].strip()
    return t


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
            self.embed: Optional[np.ndarray] = None
            self.visits = 0
            self.score_sum = 0.0
            self.data = data

            try:
                text_to_encode = " -> ".join(self.steps)
                if text_to_encode:
                    emb_list = encode([text_to_encode])
                    if len(emb_list) > 0 and emb_list[0] is not None:
                        self.embed = np.array(emb_list[0], dtype=float)
            except Exception as e:
                logger.error("Failed to encode node %d steps: %s", self.id, e)
                self.embed = None

        def score(self) -> float:
            return self.score_sum / self.visits if self.visits > 0 else 0.0

        def is_ancestor(self, potential_ancestor: "GraphOfThoughts._Node") -> bool:
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
        final_answer_format: Literal["text", "json"] = "text",
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self.llm = llm
        self.max_iters = max_iters
        self.num_branches = num_branches
        self.beam_width = beam_width
        self.merge_threshold = merge_threshold
        self.expand_prompt = expand_prompt
        self.eval_prompt = eval_prompt
        self.final_answer_format = final_answer_format

        self.max_tokens = max_tokens
        self.seed = seed

    def _parse(self, raw: str) -> List[str]:
        raw_stripped = _strip_fences(raw)
        try:
            parsed_obj = json.loads(raw_stripped)
            thought_list: Optional[List[Any]] = None

            if isinstance(parsed_obj, dict) and "thoughts" in parsed_obj:
                if isinstance(parsed_obj["thoughts"], list):
                    thought_list = parsed_obj["thoughts"]
            elif isinstance(parsed_obj, list):
                thought_list = parsed_obj
            else:
                logger.warning(
                    f"Parsed JSON is not a list or dict with 'thoughts': {type(parsed_obj)}"
                )
                return []

            if thought_list is not None:
                valid_thoughts = [
                    str(s).strip()
                    for s in thought_list
                    if isinstance(s, (str, int, float)) and str(s).strip()
                ]
                return valid_thoughts[: self.num_branches]
            else:
                return []

        except json.JSONDecodeError as e:
            logger.error(
                "Failed to parse expansion JSON: %s\nRaw Stripped: %s", e, raw_stripped[:200]
            )
            return []
        except Exception as e:
            logger.error("Unexpected error during expansion parsing: %s", e, exc_info=True)
            return []

    def _evaluate(self, steps: List[str], **kwargs) -> float:
        if not steps:
            return 0.0
        numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
        prompt = self.eval_prompt.format(steps=numbered)
        logger.debug(f"Evaluating node steps (sync) with prompt:\n{prompt}")

        try:
            result = self.llm.generate_json(
                prompt,
                response_model=EvaluationResult,
                max_tokens=kwargs.pop("max_tokens", self.max_tokens),
                seed=kwargs.pop("seed", self.seed),
                **kwargs,
            )
            if isinstance(result, EvaluationResult):
                score = float(result.score)
                normalized_score = max(0.0, min(1.0, (score - 1.0) / 9.0))
                logger.debug(
                    f"Evaluation score: {score} -> Normalized: {normalized_score:.3f}. Justification: {result.justification}"
                )
                return normalized_score
            else:
                logger.error(f"Evaluation returned unexpected type: {type(result)}")
                return 0.0
        except Exception as e:
            logger.error("Evaluation LLM call failed: %s", e, exc_info=True)
            return 0.0

    async def _evaluate_async(
        self, steps: List[str], semaphore: Optional[asyncio.Semaphore] = None, **kwargs
    ) -> float:
        if not steps:
            return 0.0
        numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
        prompt = self.eval_prompt.format(steps=numbered)
        logger.debug(f"Evaluating node steps (async) with prompt:\n{prompt}")

        local_kwargs = kwargs.copy()
        gen_args = {
            "response_model": EvaluationResult,
            "max_tokens": local_kwargs.pop("max_tokens", self.max_tokens),
            "seed": local_kwargs.pop("seed", self.seed),
            **local_kwargs,
        }

        try:
            if semaphore:
                async with semaphore:
                    result = await self.llm.generate_json_async(prompt, **gen_args)
            else:
                result = await self.llm.generate_json_async(prompt, **gen_args)

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
        if new_node.embed is None:
            logger.debug(f"Skipping similarity check for node {new_node.id} (no embedding).")
            return None

        new_norm = np.linalg.norm(new_node.embed)
        if new_norm == 0:
            logger.debug(f"Skipping similarity check for node {new_node.id} (zero norm embedding).")
            return None

        logger.debug(
            f"Checking similarity for node {new_node.id} against {len(nodes_to_check)} nodes."
        )
        for other in nodes_to_check:
            if other.id == new_node.id or other.embed is None:
                continue

            other_norm = np.linalg.norm(other.embed)
            if other_norm == 0 or new_node.is_ancestor(other):
                continue

            try:
                dot_product = np.dot(new_node.embed.flatten(), other.embed.flatten())
                sim = float(dot_product / (new_norm * other_norm))
            except ValueError as e:
                logger.warning(
                    f"Error calculating similarity between node {new_node.id} and {other.id}: {e}"
                )
                continue

            if sim > self.merge_threshold:
                logger.info(
                    f"Merging node {new_node.id} into similar node {other.id} (similarity: {sim:.3f})"
                )
                return other

        return None

    def run(self, question: str, **kwargs) -> str:
        GraphOfThoughts._Node._id_counter = 0
        root = self._Node([question])
        frontier = [root]
        all_nodes = {root.id: root}

        logger.info(
            f"Starting GoT run. Max iterations: {self.max_iters}, Beam width: {self.beam_width}"
        )

        for iter_num in range(self.max_iters):
            logger.info(f"--- GoT Iteration {iter_num + 1}/{self.max_iters} ---")
            logger.debug(f"Frontier size: {len(frontier)}. Nodes: {[n.id for n in frontier]}")
            if not frontier:
                logger.info("Frontier is empty. Stopping iterations.")
                break

            expansion_results: Dict[int, List[str]] = {}
            logger.info(f"Expanding {len(frontier)} nodes in the frontier...")
            for node in frontier:
                ctx = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(node.steps))
                prompt = self.expand_prompt.format(k=self.num_branches, ctx=ctx)
                exps: List[str] = []

                try:
                    local_kwargs = kwargs.copy()
                    gen_seed = (self.seed + iter_num + node.id) if self.seed is not None else None
                    raw = self.llm.generate(
                        prompt,
                        max_tokens=local_kwargs.pop("max_tokens", self.max_tokens),
                        seed=gen_seed,
                        **local_kwargs,
                    )
                    exps = self._parse(raw)
                    logger.debug(f"Node {node.id} expanded into {len(exps)} thoughts.")
                except Exception as e:
                    logger.error(f"Expansion failed for node {node.id}: {e}", exc_info=True)
                    exps = []

                expansion_results[node.id] = exps

            newly_added: List[GraphOfThoughts._Node] = []
            for node in frontier:
                for step in expansion_results.get(node.id, []):
                    new_node = self._Node(node.steps + [step], parents=[node])
                    similar = self._find_similar_node(new_node, list(all_nodes.values()))

                    if similar:
                        if node not in similar.parents:
                            similar.parents.append(node)
                            logger.debug(
                                f"Added node {node.id} as parent to existing node {similar.id}"
                            )
                        continue
                    else:
                        node.children.append(new_node)
                        all_nodes[new_node.id] = new_node
                        newly_added.append(new_node)
                        logger.debug(f"Added new node {new_node.id} from parent {node.id}")

            if not newly_added:
                logger.info("No new nodes were added in this iteration. Stopping.")
                break

            logger.info(f"Evaluating {len(newly_added)} newly added nodes...")
            scored_nodes: List[Tuple[float, GraphOfThoughts._Node]] = []
            for n in newly_added:
                node_score = self._evaluate(n.steps, **kwargs)
                n.visits += 1
                n.score_sum += node_score
                scored_nodes.append((n.score(), n))

            scored_nodes.sort(key=lambda x: x[0], reverse=True)
            frontier = [node for score, node in scored_nodes[: self.beam_width]]
            logger.info(
                f"Selected top {len(frontier)} nodes for next frontier (Beam Width: {self.beam_width})."
            )

            if not frontier:
                logger.info("Frontier became empty after pruning. Stopping.")
                break

        final_candidates = frontier or list(all_nodes.values())
        if not final_candidates:
            logger.error("No candidate nodes found at the end of GoT run.")
            return "Error: No reasoning paths generated."

        best_node = max(final_candidates, key=lambda n: n.score())
        logger.info(f"Selected best node: {best_node}")

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

        try:
            local_kwargs_final = kwargs.copy()
            if self.final_answer_format == "json":
                json_req = (
                    final_prompt
                    + '\n\nReturn exactly one JSON object with a single key "final_answer" whose value is the answer string.\n\nJSON Answer:'
                )
                parsed = self.llm.generate_json(
                    json_req,
                    response_model=ExtractedAnswer,
                    max_tokens=local_kwargs_final.pop("max_tokens", self.max_tokens),
                    seed=local_kwargs_final.pop("seed", self.seed),
                    **local_kwargs_final,
                )
                return parsed.final_answer.strip()
            else:
                return self.llm.generate(
                    final_prompt,
                    max_tokens=local_kwargs_final.pop("max_tokens", self.max_tokens),
                    seed=local_kwargs_final.pop("seed", self.seed),
                    **local_kwargs_final,
                ).strip()
        except Exception as e:
            logger.error("Final answer generation failed: %s", e, exc_info=True)
            return "Error generating final answer."

    async def run_async(
        self, question: str, semaphore: Optional[asyncio.Semaphore] = None, **kwargs
    ) -> str:
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

            logger.info(f"Expanding {len(frontier)} nodes in the frontier asynchronously...")

            async def expand_task(node: GraphOfThoughts._Node) -> Tuple[int, List[str]]:
                ctx = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(node.steps))
                prompt = self.expand_prompt.format(k=self.num_branches, ctx=ctx)
                exps: List[str] = []

                try:
                    local_kwargs = kwargs.copy()
                    gen_seed = (self.seed + iter_num + node.id) if self.seed is not None else None
                    gen_args = {
                        "max_tokens": local_kwargs.pop("max_tokens", self.max_tokens),
                        "seed": gen_seed,
                        **local_kwargs,
                    }
                    if semaphore:
                        async with semaphore:
                            raw = await self.llm.generate_async(prompt, **gen_args)
                    else:
                        raw = await self.llm.generate_async(prompt, **gen_args)
                    exps = self._parse(raw)
                    logger.debug(f"Node {node.id} expanded into {len(exps)} thoughts (async).")
                except Exception as e:
                    logger.error(f"Async expansion failed for node {node.id}: {e}", exc_info=True)
                    exps = []

                return node.id, exps

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
                    expansion_results[res[0]] = res[1]
                else:
                    logger.error(
                        f"Unexpected result type from expand_task for node {node_id}: {type(res)}"
                    )
                    expansion_results[node_id] = []

            newly_added: List[GraphOfThoughts._Node] = []
            for nid, steps in expansion_results.items():
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

            logger.info(f"Evaluating {len(newly_added)} newly added nodes asynchronously...")

            async def eval_task(node: GraphOfThoughts._Node) -> Tuple[int, float]:
                node_score = await self._evaluate_async(node.steps, semaphore, **kwargs)
                return node.id, node_score

            score_results = await asyncio.gather(
                *(eval_task(n) for n in newly_added), return_exceptions=True
            )

            scored_nodes: List[Tuple[float, GraphOfThoughts._Node]] = []
            processed_ids = set()
            for i, res in enumerate(score_results):
                node_id = newly_added[i].id
                processed_ids.add(node_id)
                node = all_nodes.get(node_id)
                if node is None:
                    continue

                if isinstance(res, Exception):
                    logger.error(f"Async evaluation task failed for node {node_id}: {res}")
                    node_score = 0.0
                elif isinstance(res, tuple) and len(res) == 2:
                    node_score = res[1]
                else:
                    logger.error(
                        f"Unexpected result type from eval_task for node {node_id}: {type(res)}"
                    )
                    node_score = 0.0

                node.visits += 1
                node.score_sum += node_score
                scored_nodes.append((node.score(), node))

            scored_nodes.sort(key=lambda x: x[0], reverse=True)
            frontier = [node for score, node in scored_nodes[: self.beam_width]]
            logger.info(
                f"Selected top {len(frontier)} nodes for next frontier (Async Beam Width: {self.beam_width})."
            )

            if not frontier:
                logger.info("Frontier became empty after pruning (async). Stopping.")
                break

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
            local_kwargs_final = kwargs.copy()
            final_seed = local_kwargs_final.pop("seed", self.seed)
            final_max_tokens = local_kwargs_final.pop("max_tokens", self.max_tokens)

            if self.final_answer_format == "json":
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
                return parsed.final_answer.strip()
            else:
                gen_args = {
                    "max_tokens": final_max_tokens,
                    "seed": final_seed,
                    **local_kwargs_final,
                }
                if semaphore:
                    async with semaphore:
                        return (await self.llm.generate_async(final_prompt, **gen_args)).strip()
                else:
                    return (await self.llm.generate_async(final_prompt, **gen_args)).strip()
        except Exception as e:
            logger.error("Final async answer generation failed: %s", e, exc_info=True)
            return "Error generating final async answer."
