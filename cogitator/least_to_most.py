# cogitator/least_to_most.py

import asyncio
import json
import logging
from typing import List, Optional, Tuple

from .model import BaseLLM
from .schemas import ExtractedAnswer, LTMDecomposition

logger = logging.getLogger(__name__)


class LeastToMost:
    def __init__(
        self,
        llm: BaseLLM,
        few_shot_examples: Optional[List[Tuple[str, List[str]]]] = None,
        *,
        use_json_parsing: bool = False,
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
        self.use_json_parsing = use_json_parsing
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
            result = self.llm.generate_json(prompt, response_model=LTMDecomposition)
            arr = result.subquestions or []
        except Exception as e:
            logger.error(
                "Decomposition JSON call failed for question '%s': %s", question, e, exc_info=True
            )
            err = str(e).splitlines()[0]
            tag = "JSONDecodeError" if "Invalid JSON" in str(e) else "ValidationError"
            raise ValueError(
                f"Failed to decompose question due to LLM error: {type(e).__name__}: {err} {tag}"
            ) from e

        subs = [s.strip() for s in arr if s and isinstance(s, str)]
        if not subs:
            raise ValueError("LLM returned empty subquestions list after validation.")
        return subs[: self.max_subqs]


    async def decompose_async(
        self, question: str, semaphore: Optional[asyncio.Semaphore] = None
    ) -> List[str]:
        prompt = self._build_prefix() + self.decompose_prompt_template.format(question=question)
        logger.debug("LTM Async Decompose Prompt:\n%s", prompt)
        try:
            if semaphore:
                async with semaphore:
                    result = await self.llm.generate_json_async(
                        prompt, response_model=LTMDecomposition
                    )
            else:
                result = await self.llm.generate_json_async(
                    prompt, response_model=LTMDecomposition
                )
            arr = result.subquestions or []
        except Exception as e:
            logger.error(
                "Async decomposition JSON call failed for question '%s': %s",
                question,
                e,
                exc_info=True,
            )
            err = str(e).splitlines()[0]
            tag = "JSONDecodeError" if "Invalid JSON" in str(e) else "ValidationError"
            raise ValueError(
                f"Async decomposition failed due to LLM error: {type(e).__name__}: {err} {tag}"
            ) from e

        subs = [s.strip() for s in arr if s and isinstance(s, str)]
        if not subs:
            raise ValueError("Async LLM returned empty subquestions list after validation.")
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
                if self.use_json_parsing:
                    # wrap in small JSON envelope
                    json_p = (
                        prompt
                        + '\n\nReturn exactly one JSON object with key "final_answer" whose value is the answer.\n\nJSON Answer:'
                    )
                    parsed = self.llm.generate_json(json_p, response_model=ExtractedAnswer)
                    ans = parsed.final_answer.strip()
                else:
                    ans = self.llm.generate(prompt).strip()
                if not ans:
                    ans = "[No Answer Found]"
            except Exception as e:
                logger.error("Error solving '%s': %s", sub, e, exc_info=True)
                ans = "[Error]"
            solved.append((sub, ans))
        return solved

    async def solve_async(
        self, question: str, subqs: List[str], semaphore: Optional[asyncio.Semaphore] = None
    ) -> List[Tuple[str, str]]:
        solved: List[Tuple[str, str]] = []

        async def one(i: int, sq: str, ctx: str) -> Tuple[int, str, str]:
            prompt = self.solve_prompt_template.format(context=ctx, subquestion=sq)
            logger.debug("LTM Async Solve Subquestion %d Prompt:\n%s", i + 1, prompt)
            try:
                if self.use_json_parsing:
                    json_p = (
                        prompt
                        + '\n\nReturn exactly one JSON object with key "final_answer" whose value is the answer.\n\nJSON Answer:'
                    )
                    if semaphore:
                        async with semaphore:
                            parsed = await self.llm.generate_json_async(
                                json_p, response_model=ExtractedAnswer
                            )
                    else:
                        parsed = await self.llm.generate_json_async(
                            json_p, response_model=ExtractedAnswer
                        )
                    ans = parsed.final_answer.strip()
                else:
                    if semaphore:
                        async with semaphore:
                            ans = await self.llm.generate_async(prompt)
                    else:
                        ans = await self.llm.generate_async(prompt)
                    ans = ans.strip()

                if not ans:
                    ans = "[No Answer Found]"
            except Exception:
                ans = "[Error]"
            return i, sq, ans

        for i, sq in enumerate(subqs):
            context = (
                "Previously solved:\n" + "\n".join(f"Q: {q}\nA: {a}" for q, a in solved) + "\n"
                if solved
                else "None."
            )
            idx, sqr, ansr = await one(i, sq, context)
            solved.append((sqr, ansr))

        return solved

    def answer(self, question: str) -> str:
        try:
            subs = self.decompose(question)
            solved = self.solve(question, subs)
        except Exception as e:
            return f"Error: {e}"

        steps = "\n".join(f"{i + 1}. Q: {q}\n   A: {a}" for i, (q, a) in enumerate(solved))
        prompt = self.final_answer_prompt_template.format(solved_steps=steps, question=question)
        logger.debug("LTM Final Answer Prompt:\n%s", prompt)

        try:
            if self.use_json_parsing:
                json_p = (
                    prompt
                    + '\n\nReturn exactly one JSON object with key "final_answer" whose value is the answer.\n\nJSON Answer:'
                )
                parsed = self.llm.generate_json(json_p, response_model=ExtractedAnswer)
                return parsed.final_answer.strip()
            else:
                return self.llm.generate(prompt).strip()
        except Exception as e:
            return f"Error: {e}"

    async def answer_async(
        self, question: str, semaphore: Optional[asyncio.Semaphore] = None
    ) -> str:
        try:
            subs = await self.decompose_async(question, semaphore)
            solved = await self.solve_async(question, subs, semaphore)
        except Exception as e:
            return f"Error: {e}"

        steps = "\n".join(f"{i + 1}. Q: {q}\n   A: {a}" for i, (q, a) in enumerate(solved))
        prompt = self.final_answer_prompt_template.format(solved_steps=steps, question=question)
        logger.debug("LTM Async Final Answer Prompt:\n%s", prompt)

        try:
            if self.use_json_parsing:
                json_p = (
                    prompt
                    + '\n\nReturn exactly one JSON object with key "final_answer" whose value is the answer.\n\nJSON Answer:'
                )
                if semaphore:
                    async with semaphore:
                        parsed = await self.llm.generate_json_async(
                            json_p, response_model=ExtractedAnswer
                        )
                else:
                    parsed = await self.llm.generate_json_async(
                        json_p, response_model=ExtractedAnswer
                    )
                return parsed.final_answer.strip()
            else:
                if semaphore:
                    async with semaphore:
                        ans = await self.llm.generate_async(prompt)
                else:
                    ans = await self.llm.generate_async(prompt)
                return ans.strip()
        except Exception as e:
            return f"Error: {e}"

    __call__ = answer
