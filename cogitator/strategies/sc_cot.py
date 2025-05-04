import asyncio
import logging
import re
from collections import Counter
from typing import Any, AsyncIterator, Iterator, List, Literal, Optional

from ..model import BaseLLM
from ..schemas import ExtractedAnswer

logger = logging.getLogger(__name__)


class SelfConsistency:
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

    def _extract_answer_json(self, cot: str, **kwargs) -> str:
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

    async def _extract_answer_json_async(self, cot: str, **kwargs) -> str:
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

    def extract_answer(self, cot: str, **kwargs) -> str:
        if self.internal_extraction_format == "json":
            return self._extract_answer_json(cot, **kwargs)
        return self._extract_answer_heuristic(cot)

    async def extract_answer_async(self, cot: str, **kwargs) -> str:
        if self.internal_extraction_format == "json":
            return await self._extract_answer_json_async(cot, **kwargs)
        return self._extract_answer_heuristic(cot)

    def run(self, prompt: str, **kwargs) -> str:
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
        self, prompt: str, semaphore: Optional[asyncio.Semaphore] = None, **kwargs
    ) -> str:
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

            if semaphore:
                await semaphore.acquire()
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
                if semaphore:
                    semaphore.release()

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
        raise NotImplementedError("Streaming not supported for SelfConsistency.")

    async def run_stream_async(self, prompt: str) -> AsyncIterator[str]:
        raise NotImplementedError("Streaming not supported for SelfConsistency.")
