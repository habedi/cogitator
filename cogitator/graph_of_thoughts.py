import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional

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

            try:
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
        use_json: bool = False,
    ):
        self.llm = llm
        # For raw text expansions, bypass any JSONOnlyLLM wrapper:
        self._raw_llm = getattr(llm, "_real", llm)

        self.max_iters = max_iters
        self.num_branches = num_branches
        self.beam_width = beam_width
        self.merge_threshold = merge_threshold
        self.expand_prompt = expand_prompt
        self.eval_prompt = eval_prompt
        self.use_json = use_json

    def _parse(self, raw: str) -> List[str]:
        raw_stripped = _strip_fences(raw)
        try:
            parsed_obj = json.loads(raw_stripped)
            if isinstance(parsed_obj, dict) and "thoughts" in parsed_obj:
                thought_list = parsed_obj["thoughts"]
            elif isinstance(parsed_obj, list):
                thought_list = parsed_obj
            else:
                return []
            return [
                str(s).strip()
                for s in thought_list
                if isinstance(s, (str, int, float)) and str(s).strip()
            ][: self.num_branches]
        except Exception as e:
            logger.error("Failed to parse expansion JSON: %s\n%s", e, raw_stripped[:200])
            return []

    def _evaluate(self, steps: List[str]) -> float:
        numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
        prompt = self.eval_prompt.format(steps=numbered)
        try:
            result = self.llm.generate_json(prompt, response_model=EvaluationResult)
            score = float(result.score)
            return max(0.0, min(1.0, (score - 1.0) / 9.0))
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
                    result = await self.llm.generate_json_async(
                        prompt, response_model=EvaluationResult
                    )
            else:
                result = await self.llm.generate_json_async(prompt, response_model=EvaluationResult)
            score = float(result.score)
            return max(0.0, min(1.0, (score - 1.0) / 9.0))
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
            if other_norm == 0 or new_node.is_ancestor(other):
                continue
            sim = float(
                np.dot(new_node.embed.flatten(), other.embed.flatten()) / (new_norm * other_norm)
            )
            if sim > self.merge_threshold:
                return other
        return None

    def run(self, question: str) -> str:
        GraphOfThoughts._Node._id_counter = 0
        root = self._Node([question])
        frontier = [root]
        all_nodes = {root.id: root}

        for _ in range(self.max_iters):
            expansion_results: Dict[int, List[str]] = {}

            for node in frontier:
                ctx = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(node.steps))
                prompt = self.expand_prompt.format(k=self.num_branches, ctx=ctx)

                if self.use_json:
                    # still fetch raw JSON text then json.loads
                    json_p = prompt + "\n\nReturn exactly one JSON list of strings.\n\nJSON Answer:"
                    try:
                        raw = self._raw_llm.generate(json_p)
                        arr = json.loads(_strip_fences(raw))
                        exps = [str(s).strip() for s in arr if isinstance(s, (str, int, float))]
                    except Exception as e:
                        logger.error("JSON expansion failed for node %d: %s", node.id, e)
                        exps = []
                else:
                    raw = self._raw_llm.generate(prompt)
                    exps = self._parse(raw)

                expansion_results[node.id] = exps

            newly_added = []
            for node in frontier:
                for step in expansion_results.get(node.id, []):
                    new_node = self._Node(node.steps + [step], parents=[node])
                    similar = self._find_similar_node(new_node, list(all_nodes.values()))
                    if similar:
                        if node not in similar.parents:
                            similar.parents.append(node)
                        continue
                    node.children.append(new_node)
                    all_nodes[new_node.id] = new_node
                    newly_added.append(new_node)

            if not newly_added:
                break

            scored = []
            for n in newly_added:
                s = self._evaluate(n.steps)
                n.visits += 1
                n.score_sum += s
                scored.append((n.score(), n))
            scored.sort(key=lambda x: x[0], reverse=True)
            frontier = [n for _, n in scored[: self.beam_width]]
            if not frontier:
                break

        final_candidates = frontier or list(all_nodes.values())
        best = max(final_candidates, key=lambda n: n.score())
        reasoning = best.steps[1:]
        numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(reasoning))
        final_prompt = f"Given reasoning steps:\n{numbered}\n\nAnswer the question: {question}"

        if self.use_json:
            json_req = (
                final_prompt
                + '\n\nReturn exactly one JSON object with a single key "final_answer" whose value is the answer string.\n\nJSON Answer:'
            )
            try:
                parsed = self.llm.generate_json(json_req, response_model=ExtractedAnswer)
                return parsed.final_answer.strip()
            except Exception as e:
                logger.error("Final JSON answer failed: %s", e)
                return "Error generating final answer."
        else:
            try:
                return self.llm.generate(final_prompt).strip()
            except Exception as e:
                logger.error("Final answer generation failed: %s", e)
                return "Error generating final answer."

    async def run_async(self, question: str, semaphore: Optional[asyncio.Semaphore] = None) -> str:
        GraphOfThoughts._Node._id_counter = 0
        root = self._Node([question])
        frontier = [root]
        all_nodes = {root.id: root}

        for _ in range(self.max_iters):

            async def expand_task(n):
                ctx = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(n.steps))
                prompt = self.expand_prompt.format(k=self.num_branches, ctx=ctx)

                if self.use_json:
                    json_p = prompt + "\n\nReturn exactly one JSON list of strings.\n\nJSON Answer:"
                    try:
                        if semaphore:
                            async with semaphore:
                                raw = await self._raw_llm.generate_async(json_p)
                        else:
                            raw = await self._raw_llm.generate_async(json_p)
                        arr = json.loads(_strip_fences(raw))
                        exps = [str(s).strip() for s in arr if isinstance(s, (str, int, float))]
                    except Exception as e:
                        logger.error("Async JSON expansion failed for %d: %s", n.id, e)
                        exps = []
                    return n.id, exps
                else:
                    if semaphore:
                        async with semaphore:
                            raw = await self._raw_llm.generate_async(prompt)
                    else:
                        raw = await self._raw_llm.generate_async(prompt)
                    return n.id, self._parse(raw)

            results = await asyncio.gather(*(expand_task(n) for n in frontier))
            expansion_results = dict(results)

            newly_added = []
            for nid, steps in expansion_results.items():
                parent = all_nodes[nid]
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
                break

            async def eval_task(n):
                sc = await self._evaluate_async(n.steps, semaphore)
                return n.id, sc

            scores = await asyncio.gather(*(eval_task(n) for n in newly_added))
            scored = []
            processed = set()
            for nid, sc in scores:
                node = all_nodes[nid]
                node.visits += 1
                node.score_sum += sc
                scored.append((node.score(), node))
                processed.add(nid)
            for f in frontier:
                if f.id not in processed:
                    scored.append((all_nodes[f.id].score(), f))

            scored.sort(key=lambda x: x[0], reverse=True)
            frontier = [n for _, n in scored[: self.beam_width]]
            if not frontier:
                break

        final_candidates = frontier or list(all_nodes.values())
        best = max(final_candidates, key=lambda n: n.score())
        reasoning = best.steps[1:]
        numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(reasoning))
        final_prompt = f"Given reasoning steps:\n{numbered}\n\nAnswer the question: {question}"

        if self.use_json:
            json_req = (
                final_prompt
                + '\n\nReturn exactly one JSON object with a single key "final_answer" whose value is the answer string.\n\nJSON Answer:'
            )
            try:
                if semaphore:
                    async with semaphore:
                        parsed = await self.llm.generate_json_async(
                            json_req, response_model=ExtractedAnswer
                        )
                else:
                    parsed = await self.llm.generate_json_async(
                        json_req, response_model=ExtractedAnswer
                    )
                return parsed.final_answer.strip()
            except Exception as e:
                logger.error("Final async JSON answer failed: %s", e)
                return "Error generating final async answer."
        else:
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
