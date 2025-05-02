import asyncio
import logging
from collections import Counter
from typing import Any, AsyncIterator, Iterator, List, Optional
import re

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
        use_json_parsing: bool = False,
        answer_extraction_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        **gen_kwargs: Any,
    ):
        self.llm = llm
        self.n_samples = n_samples
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop = stop
        self.use_json_parsing = use_json_parsing
        self.seed = seed
        self.gen_kwargs = gen_kwargs
        if use_json_parsing:
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
            # 1) explicit equals
            if "=" in text:
                return text.split("=", 1)[1].strip().lstrip("$").strip()
            # 2) “the answer is X”
            m0 = re.search(r"(?i)\bthe answer is\s+(\S+)", text)
            if m0:
                return m0.group(1).lstrip("$").strip()
            # 3) starts with Answer:, Ans, Final Answer:
            m1 = re.match(r"(?i)^(?:Answer|Final Answer|Ans)\b[: ]\s*(.+)$", text)
            if m1:
                return m1.group(1).strip()
            # 4) markdown heading + number, e.g. “### 123”
            m2 = re.match(r"^#+\s*([+-]?\d+(?:\.\d+)?)$", text)
            if m2:
                return m2.group(1)
            # 5) pure number
            if re.fullmatch(r"\$?[+-]?\d+(?:\.\d+)?", text):
                return text.lstrip("$")
        # fallback: last line
        return lines[-1].strip()

    def _extract_answer_json(self, cot: str) -> str:
        prompt = self.answer_extraction_prompt.format(cot=cot)
        logger.debug("Attempting JSON extraction with prompt:\n%s", prompt)
        try:
            result = self.llm.generate_json(
                prompt,
                response_model=ExtractedAnswer,
                max_tokens=self.max_tokens,
                seed=self.seed,
            )
            return str(result.final_answer).strip()
        except Exception as e:
            logger.error("JSON extraction failed: %s", e, exc_info=True)
        return self._extract_answer_heuristic(cot)

    async def _extract_answer_json_async(self, cot: str) -> str:
        prompt = self.answer_extraction_prompt.format(cot=cot)
        logger.debug("Attempting async JSON extraction with prompt:\n%s", prompt)
        try:
            result = await self.llm.generate_json_async(
                prompt,
                response_model=ExtractedAnswer,
                max_tokens=self.max_tokens,
                seed=self.seed,
            )
            return str(result.final_answer).strip()
        except Exception as e:
            logger.error("Async JSON extraction failed: %s", e, exc_info=True)
        return self._extract_answer_heuristic(cot)

    def extract_answer(self, cot: str) -> str:
        if self.use_json_parsing:
            return self._extract_answer_json(cot)
        return self._extract_answer_heuristic(cot)

    async def extract_answer_async(self, cot: str) -> str:
        if self.use_json_parsing:
            return await self._extract_answer_json_async(cot)
        return self._extract_answer_heuristic(cot)

    def run(self, prompt: str) -> str:
        answers: List[str] = []
        for i in range(self.n_samples):
            try:
                cot = self.llm.generate(
                    prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stop=self.stop,
                    seed=self.seed,
                    **self.gen_kwargs,
                )
                ans = self.extract_answer(cot)
                if ans:
                    answers.append(ans)
            except Exception:
                pass
        if not answers:
            return ""
        top_answer, _ = Counter(answers).most_common(1)[0]
        return top_answer

    async def run_async(self, prompt: str, semaphore: Optional[asyncio.Semaphore] = None) -> str:
        async def sample(i: int) -> Optional[str]:
            if semaphore:
                await semaphore.acquire()
            try:
                cot = await self.llm.generate_async(
                    prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stop=self.stop,
                    seed=self.seed,
                    **self.gen_kwargs,
                )
                return await self.extract_answer_async(cot)
            except Exception:
                return None
            finally:
                if semaphore:
                    semaphore.release()

        results = await asyncio.gather(*(sample(i) for i in range(self.n_samples)))
        answers = [a for a in results if a]
        if not answers:
            return ""
        top_answer, _ = Counter(answers).most_common(1)[0]
        return top_answer

    def run_stream(self, prompt: str) -> Iterator[str]:
        raise NotImplementedError("Streaming not supported for SelfConsistency.")

    async def run_stream_async(self, prompt: str) -> AsyncIterator[str]:
        raise NotImplementedError("Streaming not supported for SelfConsistency.")

    __call__ = run
