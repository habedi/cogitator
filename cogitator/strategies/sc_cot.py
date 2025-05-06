"""Implements the Self-Consistency Chain-of-Thought (SC-CoT) strategy."""

import asyncio
import logging
import re
from collections import Counter
from typing import Any, AsyncIterator, Iterator, List, Literal, Optional

from ..model import BaseLLM
from ..schemas import ExtractedAnswer

logger = logging.getLogger(__name__)


class SelfConsistency:
    """Implements the Self-Consistency Chain-of-Thought (SC-CoT) strategy.

    Self-Consistency improves CoT prompting by generating multiple diverse
    reasoning paths (using sampling with temperature > 0) and then selecting
    the most consistent answer among the paths via majority voting.

    Reference:
        Wang et al. (v4; 2023) "Self-Consistency Improves Chain of Thought Reasoning in Language Models".
        https://arxiv.org/abs/2203.11171
    """

    def __init__(
        self,
        llm: BaseLLM,
        n_samples: int = 10,
        temperature: float = 0.8,
        max_tokens: int = 256,
        stop: Optional[List[str]] = None,
        internal_extraction_format: Literal["heuristic", "json"] = "heuristic",
        answer_extraction_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        **gen_kwargs: Any,
    ) -> None:
        """Initializes the SelfConsistency strategy handler.

        Args:
            llm: The language model instance.
            n_samples: The number of reasoning paths (samples) to generate.
            temperature: Sampling temperature for generating diverse paths. Should be > 0.
            max_tokens: Maximum tokens for each generated reasoning path.
            stop: Optional stop sequences for LLM generation.
            internal_extraction_format: Method for extracting the final answer from
                                        each CoT path ('heuristic' or 'json').
            answer_extraction_prompt: Prompt template used only if `internal_extraction_format`
                                      is 'json'. Must include {cot}. Expects JSON output
                                      matching ExtractedAnswer schema.
            seed: Base random seed for LLM sampling (each sample may use seed + i).
            **gen_kwargs: Additional default keyword arguments for LLM generation calls.
        """
        self.llm = llm
        self.n_samples = n_samples
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop = stop
        self.internal_extraction_format = internal_extraction_format
        self.seed = seed
        self.gen_kwargs = gen_kwargs

        if self.internal_extraction_format == "json":
            self.answer_extraction_prompt = (
                answer_extraction_prompt
                or "Analyze the following reasoning chain and extract the final numerical or short answer. "
                "Return the result as a JSON object with a single key 'final_answer' containing the answer as a string.\n\n"
                "Reasoning Chain:\n{cot}\n\nJSON Answer:"
            )
        else:
            self.answer_extraction_prompt = None

    def _extract_answer_heuristic(self, cot: str) -> str:
        """Extracts the final answer from a CoT string using heuristics.

        Searches for common patterns like "answer is X", lines starting with "Answer:",
        numeric lines, etc., working from the end of the CoT string upwards.

        Args:
            cot: The Chain-of-Thought reasoning string.

        Returns:
            The extracted answer string, or the last line as a fallback.
        """
        lines = cot.strip().splitlines()
        for line in reversed(lines):
            text = line.strip().rstrip(".")
            if "=" in text:
                parts = text.split("=", 1)
                if len(parts) > 1:
                    answer = parts[1].strip().lstrip("$").strip()
                    logger.debug(f"Heuristically extracted answer (equals): '{answer}'")
                    return answer
            m0 = re.search(r"(?i)\bthe answer is\s+(\S+)", text)
            if m0:
                answer = m0.group(1).lstrip("$").strip()
                logger.debug(f"Heuristically extracted answer (the answer is): '{answer}'")
                return answer
            m1 = re.match(r"(?i)^(?:Answer|Final Answer|Ans)\b[: ]\s*(.+)$", text)
            if m1:
                answer = m1.group(1).strip()
                logger.debug(f"Heuristically extracted answer (Prefix): '{answer}'")
                return answer
            m2 = re.match(r"^#+\s*([+-]?\d+(?:\.\d+)?)$", text)
            if m2:
                answer = m2.group(1)
                logger.debug(f"Heuristically extracted answer (Header): '{answer}'")
                return answer
            m3 = re.match(r"^\*{1,2}A[: ]\s*(.+?)\*{0,2}$", text, re.IGNORECASE)
            if m3:
                answer = m3.group(1).strip()
                logger.debug(f"Heuristically extracted answer (Markdown A:): '{answer}'")
                return answer
            m4 = re.search(r":\s*([+-]?\d+(?:\.\d+)?|[A-Za-z]+)\s*$", text)
            if m4:
                answer = m4.group(1).strip()
                logger.debug(f"Heuristically extracted answer (Colon End): '{answer}'")
                return answer
            if re.fullmatch(r"\$?[+-]?\d+(?:\.\d+)?", text):
                answer = text.lstrip("$")
                logger.debug(f"Heuristically extracted answer (Numeric Line): '{answer}'")
                return answer
        fallback_answer = lines[-1].strip() if lines else ""
        logger.debug(f"Heuristically extracted answer (Fallback): '{fallback_answer}'")
        return fallback_answer

    def _extract_answer_json(self, cot: str, **kwargs: Any) -> str:
        """Extracts the final answer using an LLM call with JSON parsing.

        Uses the `answer_extraction_prompt` and expects a JSON response matching
        the `ExtractedAnswer` schema. Falls back to heuristic extraction on failure.

        Args:
            cot: The Chain-of-Thought reasoning string.
            **kwargs: Additional arguments passed to the LLM `generate_json` call.

        Returns:
            The extracted answer string.
        """
        if not self.answer_extraction_prompt:
            logger.warning("JSON extraction requested but prompt is not configured.")
            return self._extract_answer_heuristic(cot)

        prompt = self.answer_extraction_prompt.format(cot=cot)
        logger.debug("Attempting JSON extraction with prompt:\n%s", prompt)
        try:
            local_kwargs = kwargs.copy()
            result = self.llm.generate_json(
                prompt,
                response_model=ExtractedAnswer,
                max_tokens=local_kwargs.pop("max_tokens", self.max_tokens),
                seed=local_kwargs.pop("seed", self.seed),
                **local_kwargs,
            )
            answer = str(result.final_answer).strip()
            logger.debug(f"JSON extracted answer: '{answer}'")
            return answer
        except Exception as e:
            logger.error("JSON extraction failed: %s", e, exc_info=True)
        logger.warning("JSON extraction failed, falling back to heuristic.")
        return self._extract_answer_heuristic(cot)

    async def _extract_answer_json_async(self, cot: str, **kwargs: Any) -> str:
        """Asynchronously extracts the final answer using an LLM call with JSON parsing.

        Similar to `_extract_answer_json` but uses async LLM calls.

        Args:
            cot: The Chain-of-Thought reasoning string.
            **kwargs: Additional arguments passed to the async LLM `generate_json_async` call.

        Returns:
            The extracted answer string.
        """
        if not self.answer_extraction_prompt:
            logger.warning("Async JSON extraction requested but prompt is not configured.")
            return self._extract_answer_heuristic(cot)

        prompt = self.answer_extraction_prompt.format(cot=cot)
        logger.debug("Attempting async JSON extraction with prompt:\n%s", prompt)
        try:
            local_kwargs = kwargs.copy()
            result = await self.llm.generate_json_async(
                prompt,
                response_model=ExtractedAnswer,
                max_tokens=local_kwargs.pop("max_tokens", self.max_tokens),
                seed=local_kwargs.pop("seed", self.seed),
                **local_kwargs,
            )
            answer = str(result.final_answer).strip()
            logger.debug(f"Async JSON extracted answer: '{answer}'")
            return answer
        except Exception as e:
            logger.error("Async JSON extraction failed: %s", e, exc_info=True)
        logger.warning("Async JSON extraction failed, falling back to heuristic.")
        return self._extract_answer_heuristic(cot)

    def extract_answer(self, cot: str, **kwargs: Any) -> str:
        """Extracts the final answer from a CoT string based on the configured method.

        Delegates to either `_extract_answer_heuristic` or `_extract_answer_json`.

        Args:
            cot: The Chain-of-Thought reasoning string.
            **kwargs: Arguments passed to the underlying extraction method (if JSON).

        Returns:
            The extracted answer string.
        """
        if self.internal_extraction_format == "json":
            return self._extract_answer_json(cot, **kwargs)
        return self._extract_answer_heuristic(cot)

    async def extract_answer_async(self, cot: str, **kwargs: Any) -> str:
        """Asynchronously extracts the final answer based on the configured method.

        Delegates to `_extract_answer_heuristic` or `_extract_answer_json_async`.

        Args:
            cot: The Chain-of-Thought reasoning string.
            **kwargs: Arguments passed to the underlying async extraction method (if JSON).

        Returns:
            The extracted answer string.
        """
        if self.internal_extraction_format == "json":
            return await self._extract_answer_json_async(cot, **kwargs)
        return self._extract_answer_heuristic(cot)

    def run(self, prompt: str, **kwargs: Any) -> str:
        """Executes the Self-Consistency strategy.

        Generates `n_samples` reasoning paths using the LLM with the specified
        temperature. Extracts the final answer from each path and returns the
        most frequent answer (majority vote).

        Args:
            prompt: The input prompt for the LLM.
            **kwargs: Additional arguments passed to the LLM generation and
                      answer extraction calls.

        Returns:
            The most consistent answer string among the generated paths, or an
            empty string if no valid answers are generated.
        """
        answers: List[str] = []
        combined_kwargs = {**self.gen_kwargs, **kwargs}

        for i in range(self.n_samples):
            try:
                iter_seed = (self.seed + i) if self.seed is not None else None
                current_gen_kwargs = combined_kwargs.copy()
                cot = self.llm.generate(
                    prompt,
                    temperature=current_gen_kwargs.pop("temperature", self.temperature),
                    max_tokens=current_gen_kwargs.pop("max_tokens", self.max_tokens),
                    stop=current_gen_kwargs.pop("stop", self.stop),
                    seed=iter_seed,
                    **current_gen_kwargs,
                )
                logger.debug(f"Raw CoT sample {i}: {cot}")
                ans = self.extract_answer(cot, **kwargs)
                if ans:
                    answers.append(ans)
                else:
                    logger.debug(f"Sample {i} produced empty answer after extraction.")
            except Exception as e:
                logger.error(f"Error during SC sample {i}: {e}", exc_info=True)

        if not answers:
            logger.warning("SelfConsistency generated no valid answers.")
            return ""

        try:
            count = Counter(answers)
            top_answer, _ = count.most_common(1)[0]
            logger.debug(f"SelfConsistency vote counts: {count}")
            return top_answer
        except IndexError:
            logger.error("Could not determine most common answer despite having answers.")
            return ""

    async def run_async(
        self, prompt: str, semaphore: Optional[asyncio.Semaphore] = None, **kwargs: Any
    ) -> str:
        """Asynchronously executes the Self-Consistency strategy.

        Generates `n_samples` reasoning paths concurrently using async LLM calls.
        Extracts answers asynchronously and returns the majority vote answer.

        Args:
            prompt: The input prompt for the LLM.
            semaphore: Optional asyncio.Semaphore to limit concurrent LLM calls.
            **kwargs: Additional arguments passed to the async LLM generation and
                      answer extraction calls.

        Returns:
            The most consistent answer string, or an empty string if none are generated.
        """
        combined_kwargs = {**self.gen_kwargs, **kwargs}

        async def sample(i: int) -> Optional[str]:
            sample_kwargs = combined_kwargs.copy()
            iter_seed = (self.seed + i) if self.seed is not None else None
            gen_args = {
                "temperature": sample_kwargs.pop("temperature", self.temperature),
                "max_tokens": sample_kwargs.pop("max_tokens", self.max_tokens),
                "stop": sample_kwargs.pop("stop", self.stop),
                "seed": iter_seed,
                **sample_kwargs,
            }
            extraction_kwargs = kwargs.copy()

            task_semaphore = semaphore
            if task_semaphore:
                await task_semaphore.acquire()
            try:
                cot = await self.llm.generate_async(prompt, **gen_args)
                logger.debug(f"Raw async CoT sample {i}: {cot}")
                ans = await self.extract_answer_async(cot, **extraction_kwargs)
                if not ans:
                    logger.debug(f"Async sample {i} produced empty answer after extraction.")
                return ans
            except Exception as e:
                logger.error(f"Error during async SC sample {i}: {e}", exc_info=True)
                return None
            finally:
                if task_semaphore:
                    task_semaphore.release()

        results = await asyncio.gather(*(sample(i) for i in range(self.n_samples)))
        answers = [a for a in results if a is not None and a != ""]
        if not answers:
            logger.warning("SelfConsistency (async) generated no valid answers.")
            return ""

        try:
            count = Counter(answers)
            top_answer, _ = count.most_common(1)[0]
            logger.debug(f"SelfConsistency async vote counts: {count}")
            return top_answer
        except IndexError:
            logger.error("Could not determine most common async answer despite having answers.")
            return ""

    def run_stream(self, prompt: str) -> Iterator[str]:
        """Streaming is not supported for Self-Consistency."""
        raise NotImplementedError("Streaming not supported for SelfConsistency.")

    async def run_stream_async(self, prompt: str) -> AsyncIterator[str]:
        """Streaming is not supported for Self-Consistency."""
        raise NotImplementedError("Streaming not supported for SelfConsistency.")
