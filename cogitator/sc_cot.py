import asyncio
import logging
import re
from collections import Counter
from typing import Any, AsyncIterator, Iterator, List, Literal, Optional

from .model import BaseLLM
from .schemas import ExtractedAnswer

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
                return text.split("=", 1)[1].strip().lstrip("$").strip()
            m0 = re.search(r"(?i)\bthe answer is\s+(\S+)", text)
            if m0:
                return m0.group(1).lstrip("$").strip()
            m1 = re.match(r"(?i)^(?:Answer|Final Answer|Ans)\b[: ]\s*(.+)$", text)
            if m1:
                return m1.group(1).strip()
            m2 = re.match(r"^#+\s*([+-]?\d+(?:\.\d+)?)$", text)
            if m2:
                return m2.group(1)
            if re.fullmatch(r"\$?[+-]?\d+(?:\.\d+)?", text):
                return text.lstrip("$")
        return lines[-1].strip() if lines else ""

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
            return str(result.final_answer).strip()
        except Exception as e:
            logger.error("JSON extraction failed: %s", e, exc_info=True)
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
            return str(result.final_answer).strip()
        except Exception as e:
            logger.error("Async JSON extraction failed: %s", e, exc_info=True)
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
        # Combine instance defaults with call-specific kwargs
        combined_kwargs = {**self.gen_kwargs, **kwargs}

        for i in range(self.n_samples):
            try:
                # Pass temperature, max_tokens etc. from combined kwargs or defaults
                iter_seed = (self.seed + i) if self.seed is not None else None
                cot = self.llm.generate(
                    prompt,
                    temperature=combined_kwargs.pop("temperature", self.temperature),
                    max_tokens=combined_kwargs.pop("max_tokens", self.max_tokens),
                    stop=combined_kwargs.pop("stop", self.stop),
                    seed=iter_seed,
                    **combined_kwargs,  # Pass remaining specific args
                )
                ans = self.extract_answer(
                    cot, **kwargs
                )  # Pass kwargs for potential JSON extraction
                if ans:
                    answers.append(ans)
            except Exception as e:
                logger.error(f"Error during SC sample {i}: {e}", exc_info=True)

        if not answers:
            logger.warning("SelfConsistency generated no valid answers.")
            return ""
        top_answer, _ = Counter(answers).most_common(1)[0]
        return top_answer

    async def run_async(
        self, prompt: str, semaphore: Optional[asyncio.Semaphore] = None, **kwargs
    ) -> str:
        # Combine instance defaults with call-specific kwargs
        combined_kwargs = {**self.gen_kwargs, **kwargs}

        async def sample(i: int) -> Optional[str]:
            # Capture kwargs for this specific sample call
            sample_kwargs = combined_kwargs.copy()
            iter_seed = (self.seed + i) if self.seed is not None else None
            gen_args = {
                "temperature": sample_kwargs.pop("temperature", self.temperature),
                "max_tokens": sample_kwargs.pop("max_tokens", self.max_tokens),
                "stop": sample_kwargs.pop("stop", self.stop),
                "seed": iter_seed,
                **sample_kwargs,  # Pass remaining specific args
            }
            extraction_kwargs = kwargs.copy()  # Use original kwargs for extraction

            if semaphore:
                await semaphore.acquire()
            try:
                cot = await self.llm.generate_async(prompt, **gen_args)
                # Pass extraction_kwargs for potential JSON extraction config
                return await self.extract_answer_async(cot, **extraction_kwargs)
            except Exception as e:
                logger.error(f"Error during async SC sample {i}: {e}", exc_info=True)
                return None
            finally:
                if semaphore:
                    semaphore.release()

        results = await asyncio.gather(*(sample(i) for i in range(self.n_samples)))
        answers = [a for a in results if a]
        if not answers:
            logger.warning("SelfConsistency (async) generated no valid answers.")
            return ""
        top_answer, _ = Counter(answers).most_common(1)[0]
        return top_answer

    def run_stream(self, prompt: str) -> Iterator[str]:
        raise NotImplementedError("Streaming not supported for SelfConsistency.")

    async def run_stream_async(self, prompt: str) -> AsyncIterator[str]:
        raise NotImplementedError("Streaming not supported for SelfConsistency.")
