import asyncio
import logging
import re
from collections import Counter
from typing import Any, AsyncIterator, Iterator, List, Optional, Tuple

from .model import BaseLLM

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
        **gen_kwargs: Any,
    ):
        self.llm = llm
        self.n_samples = n_samples
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop = stop
        self.use_json_parsing = use_json_parsing

        if use_json_parsing:
            self.answer_extraction_prompt = (
                answer_extraction_prompt
                or "Analyze the following reasoning chain and extract the final numerical or short answer. "
                "Return the result as a JSON object with a single key 'final_answer' containing the answer as a string.\n\n"
                "Reasoning Chain:\n{cot}\n\nJSON Answer:"
            )
        else:
            self.answer_extraction_prompt = None

        self.gen_kwargs = gen_kwargs

    def _extract_answer_heuristic(self, cot: str) -> str:
        if not cot:
            return ""

        markers = [r"(?:Final Answer|The final answer is)\s*:?\s*(.*)", r"###\s*(.*)"]
        for pattern in markers:
            m = re.search(pattern, cot, re.IGNORECASE | re.MULTILINE)
            if m and m.group(1).strip():
                extracted = m.group(1).strip(" .,:;")
                logger.debug("Heuristic extracted via marker: '%s'", extracted)
                return extracted

        lines = [line.strip() for line in cot.splitlines() if line.strip()]
        if not lines:
            logger.debug("Heuristic found no non-empty lines, returning original: '%.50s'", cot)
            return cot.strip()
        last = lines[-1]

        for prefix in ["Therefore", "Answer", "Ans", "The answer is"]:
            m = re.match(rf"^{re.escape(prefix)}\s*:?\s*(.*)", last, re.IGNORECASE)
            if m and m.group(1).strip():
                extracted = m.group(1).strip(" .,:;")
                logger.debug("Heuristic extracted via prefix '%s': '%s'", prefix, extracted)
                return extracted

        if "=" in last:
            parts = last.split("=", 1)
            if len(parts) > 1 and parts[1].strip():
                extracted = parts[1].strip(" .,:;")
                logger.debug("Heuristic extracted via '=' sign: '%s'", extracted)
                return extracted

        # Use re.match with anchors to check if the *entire* last line is essentially a number
        m_last_line_num = re.match(r"^\s*[\$€£]?\s*([-+]?\d*\.?\d+)\s*$", last)
        if m_last_line_num and m_last_line_num.group(1) != ".":
            extracted = m_last_line_num.group(1)
            logger.debug("Heuristic extracted via full numeric match on last line: '%s'", extracted)
            return extracted

        # Ultimate fallback: return the last line if nothing else worked
        logger.debug("Heuristic fallback to last line (default): '%s'", last)
        return last

    def _extract_answer_json(self, cot: str) -> str:
        if not self.answer_extraction_prompt:
            raise ValueError("JSON parsing enabled but no extraction prompt defined.")
        prompt = self.answer_extraction_prompt.format(cot=cot)
        logger.debug("Attempting JSON extraction with prompt:\n%s", prompt)
        try:
            result = self.llm.generate_json(prompt)
            logger.debug("JSON extraction result: %s", result)
            if isinstance(result, dict) and "final_answer" in result:
                extracted = str(result["final_answer"]).strip()
                logger.debug("JSON extracted answer: '%s'", extracted)
                return extracted
            logger.warning("JSON extraction unexpected format: %s", result)
        except Exception as e:
            logger.error("JSON extraction failed: %s", e, exc_info=True)

        logger.debug("Falling back to heuristic extraction after JSON failure.")
        return self._extract_answer_heuristic(cot)

    async def _extract_answer_json_async(self, cot: str) -> str:
        if not self.answer_extraction_prompt:
            raise ValueError("JSON parsing enabled but no extraction prompt defined.")
        prompt = self.answer_extraction_prompt.format(cot=cot)
        logger.debug("Attempting async JSON extraction with prompt:\n%s", prompt)
        try:
            result = await self.llm.generate_json_async(prompt)
            logger.debug("Async JSON extraction result: %s", result)
            if isinstance(result, dict) and "final_answer" in result:
                extracted = str(result["final_answer"]).strip()
                logger.debug("Async JSON extracted answer: '%s'", extracted)
                return extracted
            logger.warning("Async JSON extraction unexpected format: %s", result)
        except Exception as e:
            logger.error("Async JSON extraction failed: %s", e, exc_info=True)

        logger.debug("Falling back to heuristic extraction after async JSON failure.")
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
        raw_cots: List[str] = []
        for i in range(self.n_samples):
            try:
                cot = self.llm.generate(
                    prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stop=self.stop,
                    **self.gen_kwargs,
                )
                raw_cots.append(cot)
                logger.debug("Sample %d Raw CoT:\n---\n%s\n---", i + 1, cot)
                ans = self.extract_answer(cot)
                logger.debug("Sample %d Extracted Answer: '%s'", i + 1, ans)
                if ans:
                    answers.append(ans)
                else:
                    logger.debug("Sample %d produced empty answer after extraction.", i + 1)
            except Exception as e:
                logger.error("Sample %d generation failed: %s", i + 1, e, exc_info=True)

        if not answers:
            logger.warning("No valid answers extracted after %d samples.", self.n_samples)
            return ""

        counts = Counter(answers)
        top_answer, _ = counts.most_common(1)[0]
        logger.info("SelfConsistency selected '%s' from %s", top_answer, counts)
        return top_answer

    async def run_async(self, prompt: str, semaphore: Optional[asyncio.Semaphore] = None) -> str:
        async def sample(idx: int) -> Optional[Tuple[str, str]]:
            try:
                local_kwargs = {
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "stop": self.stop,
                    **self.gen_kwargs,
                }
                if semaphore:
                    async with semaphore:
                        cot = await self.llm.generate_async(prompt, **local_kwargs)
                else:
                    cot = await self.llm.generate_async(prompt, **local_kwargs)

                logger.debug("Async Sample %d Raw CoT:\n---\n%s\n---", idx + 1, cot)
                ans = await self.extract_answer_async(cot)
                logger.debug("Async Sample %d Extracted Answer: '%s'", idx + 1, ans)
                if not ans:
                    logger.debug("Async sample %d produced empty answer after extraction.", idx + 1)
                return cot, ans
            except Exception as e:
                logger.error("Async sample %d failed: %s", idx + 1, e, exc_info=True)
                return None

        tasks = [sample(i) for i in range(self.n_samples)]
        results = await asyncio.gather(*tasks)

        answers = []
        for res in results:
            if res is not None:
                _, ans = res
                if ans is not None and ans.strip():
                    answers.append(ans)

        if not answers:
            logger.warning("No valid answers extracted async after %d samples.", self.n_samples)
            return ""

        counts = Counter(answers)
        top_answer, _ = counts.most_common(1)[0]
        logger.info("Async SelfConsistency selected '%s' from %s", top_answer, counts)
        return top_answer

    def run_stream(self, prompt: str) -> Iterator[str]:
        raise NotImplementedError("Streaming not supported for SelfConsistency.")

    async def run_stream_async(self, prompt: str) -> AsyncIterator[str]:
        raise NotImplementedError("Streaming not supported for SelfConsistency.")

    __call__ = run
