import asyncio
import json
import logging
from typing import List, Optional, Tuple

from .model import BaseLLM

logger = logging.getLogger(__name__)


class LeastToMost:
    def __init__(
        self,
        llm: BaseLLM,
        few_shot_examples: Optional[List[Tuple[str, List[str]]]] = None,
        decompose_prompt_template: str = (
            "Decompose the main question into a sequence of simpler subquestions "
            "that must be answered sequentially to solve the main question. "
            "Return the subquestions as a JSON list of strings.\n\n"
            "Main Question: {question}\n\nJSON Subquestions:"
        ),
        solve_prompt_template: str = (
            "Previous Context:\n{context}\n\n"
            "Current Subquestion: {subquestion}\n\n"
            "Answer the current subquestion using the context if necessary. "
            "Provide only the answer to the subquestion.\nAnswer:"
        ),
        final_answer_prompt_template: str = (
            "Based on the following sequential subquestions and their answers, "
            "answer the original main question.\n\n"
            "Subquestions and Answers:\n{solved_steps}\n"
            "Original Main Question: {question}\n\nFinal Answer:"
        ),
        max_subqs: int = 10,
    ):
        self.llm = llm
        self.max_subqs = max_subqs

        if few_shot_examples is None:
            self.examples = [
                (
                    "There are 3 red balls and 4 blue balls in a bag. How many balls are there in total?",
                    [
                        "How many red balls are there?",
                        "How many blue balls are there?",
                        "What is the total number of balls?",
                    ],
                ),
                (
                    "Sarah has 5 apples and gives 2 to Tom. How many apples does she have left?",
                    [
                        "How many apples did Sarah start with?",
                        "How many apples did she give away?",
                        "How many apples remain with Sarah?",
                    ],
                ),
            ]
        else:
            self.examples = few_shot_examples

        self.decompose_prompt_template = decompose_prompt_template
        self.solve_prompt_template = solve_prompt_template
        self.final_answer_prompt_template = final_answer_prompt_template

    def _build_prefix(self) -> str:
        prefix = ""
        for ex_q, ex_subs in self.examples:
            prefix += f"Main Question: {ex_q}\nJSON Subquestions: {json.dumps(ex_subs)}\n\n"
        return prefix

    def decompose(self, question: str) -> List[str]:
        prompt = self._build_prefix() + self.decompose_prompt_template.format(question=question)
        logger.debug("LTM Decompose Prompt:\n%s", prompt)
        try:
            arr = self.llm.generate_json(prompt)
            logger.debug("LTM Decompose Raw Result: %s", arr)
        except Exception as e:
            logger.error(
                "Decomposition JSON call failed for question '%s': %s", question, e, exc_info=True
            )
            raise ValueError(f"Failed to decompose question due to LLM error: {e}") from e

        if not isinstance(arr, list):
            raise ValueError(f"Decomposition LLM response was not a list, got: {type(arr)}")
        subs = [str(s).strip() for s in arr if isinstance(s, (str, int, float)) and str(s).strip()]
        if not subs:
            raise ValueError("LLM returned empty or invalid subquestions list after parsing.")
        return subs[: self.max_subqs]

    async def decompose_async(
        self, question: str, semaphore: Optional[asyncio.Semaphore] = None
    ) -> List[str]:
        prompt = self._build_prefix() + self.decompose_prompt_template.format(question=question)
        logger.debug("LTM Async Decompose Prompt:\n%s", prompt)
        try:
            if semaphore:
                async with semaphore:
                    arr = await self.llm.generate_json_async(prompt)
            else:
                arr = await self.llm.generate_json_async(prompt)
            logger.debug("LTM Async Decompose Raw Result: %s", arr)
        except Exception as e:
            logger.error(
                "Async decomposition JSON call failed for question '%s': %s",
                question,
                e,
                exc_info=True,
            )
            raise ValueError(f"Async decomposition failed due to LLM error: {e}") from e

        if not isinstance(arr, list):
            raise ValueError(f"Async decomposition LLM response was not a list, got: {type(arr)}")
        subs = [str(s).strip() for s in arr if isinstance(s, (str, int, float)) and str(s).strip()]
        if not subs:
            raise ValueError("Async LLM returned empty or invalid subquestions list after parsing.")
        return subs[: self.max_subqs]

    def solve(self, question: str, subqs: List[str]) -> List[Tuple[str, str]]:
        solved: List[Tuple[str, str]] = []
        for i, sub in enumerate(subqs):
            context = (
                "Previously solved:\n" + "\n".join(f"Q: {q}\nA: {a}" for q, a in solved) + "\n"
                if solved
                else "None."
            )
            prompt = self.solve_prompt_template.format(context=context, subquestion=sub)
            logger.debug("LTM Solve Subquestion %d Prompt:\n%s", i + 1, prompt)
            try:
                ans = self.llm.generate(prompt).strip()
                logger.debug("LTM Solve Subquestion %d Raw Answer: '%s'", i + 1, ans)
                if not ans:
                    logger.warning("Empty answer returned for subquestion: %s", sub)
                    ans = "[No Answer Found]"
            except Exception as e:
                logger.error("Error solving subquestion '%s': %s", sub, e, exc_info=True)
                ans = "[Error Solving]"
            solved.append((sub, ans))
        return solved

    async def solve_async(
        self, question: str, subqs: List[str], semaphore: Optional[asyncio.Semaphore] = None
    ) -> List[Tuple[str, str]]:
        solved: List[Tuple[str, str]] = []
        tasks = []

        async def solve_single_sub(sub_idx: int, sub_q: str, current_context: str):
            prompt = self.solve_prompt_template.format(context=current_context, subquestion=sub_q)
            logger.debug("LTM Async Solve Subquestion %d Prompt:\n%s", sub_idx + 1, prompt)
            try:
                if semaphore:
                    async with semaphore:
                        ans = await self.llm.generate_async(prompt)
                else:
                    ans = await self.llm.generate_async(prompt)
                logger.debug("LTM Async Solve Subquestion %d Raw Answer: '%s'", sub_idx + 1, ans)
                ans_stripped = ans.strip()
                if not ans_stripped:
                    logger.warning("Empty answer returned for async subquestion: %s", sub_q)
                    ans_stripped = "[No Answer Found]"
                return sub_idx, sub_q, ans_stripped
            except Exception as e:
                logger.error("Async error solving subquestion '%s': %s", sub_q, e, exc_info=True)
                return sub_idx, sub_q, "[Error Solving Async]"

        current_context_base = "Previously solved:\n"
        # IMPORTANT: Running solve_async sequentially to preserve context dependency
        for i, sub in enumerate(subqs):
            context_for_task = (
                current_context_base + "\n".join(f"Q: {q}\nA: {a}" for q, a in solved) + "\n"
                if solved
                else "None."
            )
            sub_idx, sub_q_res, ans_res = await solve_single_sub(i, sub, context_for_task)
            # Check index just in case, though it should match i
            if sub_idx == i:
                solved.append((sub_q_res, ans_res))
            else:
                logger.error(
                    "Subquestion index mismatch during async solve: expected %d, got %d", i, sub_idx
                )
                # Decide how to handle - skip? add placeholder? For now, add placeholder
                solved.append((sub, "[Error - Index Mismatch]"))

        return solved

    def answer(self, question: str) -> str:
        try:
            subqs = self.decompose(question)
            logger.debug("LTM Decomposed Subquestions for '%s': %s", question, subqs)
        except ValueError as e:
            logger.error(
                "Decomposition failed for '%s': %s", question, e, exc_info=False
            )  # Don't need full stack trace here
            return f"Error: could not decompose question ({e})"
        except Exception as e:
            logger.error(
                "Unexpected error during decomposition for '%s': %s", question, e, exc_info=True
            )
            return f"Error: unexpected issue during decomposition ({e})"

        if not subqs:
            logger.error("No subquestions generated for '%s'", question)
            return "Error: no subquestions generated."

        try:
            solved = self.solve(question, subqs)
            logger.debug("LTM Solved Steps for '%s': %s", question, solved)
        except Exception as e:
            logger.error("Unexpected error during solving for '%s': %s", question, e, exc_info=True)
            return f"Error: unexpected issue during solving ({e})"

        steps = "\n".join(f"{i + 1}. Q: {q}\n   A: {a}" for i, (q, a) in enumerate(solved))
        prompt = self.final_answer_prompt_template.format(solved_steps=steps, question=question)
        logger.debug("LTM Final Answer Prompt:\n%s", prompt)
        try:
            final_ans = self.llm.generate(prompt).strip()
            logger.debug("LTM Final Raw Answer: '%s'", final_ans)
            return final_ans
        except Exception as e:
            logger.error("Final answer generation failed for '%s': %s", question, e, exc_info=True)
            return f"Error: could not generate final answer ({e})"

    async def answer_async(
        self, question: str, semaphore: Optional[asyncio.Semaphore] = None
    ) -> str:
        try:
            subqs = await self.decompose_async(question, semaphore)
            logger.debug("LTM Async Decomposed Subquestions for '%s': %s", question, subqs)
        except ValueError as e:
            logger.error("Async decomposition failed for '%s': %s", question, e, exc_info=False)
            return f"Error: could not decompose question async ({e})"
        except Exception as e:
            logger.error(
                "Unexpected error during async decomposition for '%s': %s",
                question,
                e,
                exc_info=True,
            )
            return f"Error: unexpected issue during async decomposition ({e})"

        if not subqs:
            logger.error("No async subquestions generated for '%s'", question)
            return "Error: no async subquestions generated."

        try:
            solved = await self.solve_async(question, subqs, semaphore)
            logger.debug("LTM Async Solved Steps for '%s': %s", question, solved)
        except Exception as e:
            logger.error(
                "Unexpected error during async solving for '%s': %s", question, e, exc_info=True
            )
            return f"Error: unexpected issue during async solving ({e})"

        steps = "\n".join(f"{i + 1}. Q: {q}\n   A: {a}" for i, (q, a) in enumerate(solved))
        prompt = self.final_answer_prompt_template.format(solved_steps=steps, question=question)
        logger.debug("LTM Async Final Answer Prompt:\n%s", prompt)
        try:
            if semaphore:
                async with semaphore:
                    ans = await self.llm.generate_async(prompt)
            else:
                ans = await self.llm.generate_async(prompt)
            final_ans = ans.strip()
            logger.debug("LTM Async Final Raw Answer: '%s'", final_ans)
            return final_ans
        except Exception as e:
            logger.error(
                "Final async answer generation failed for '%s': %s", question, e, exc_info=True
            )
            return f"Error: could not generate final async answer ({e})"

    __call__ = answer
