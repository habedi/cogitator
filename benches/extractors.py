import logging
import re
from typing import Optional

logger_extractors = logging.getLogger("benchmark_extractors")


def _extract_label_heuristic(text: str) -> Optional[str]:
    text = str(text).strip()
    lines = text.strip().splitlines()

    mcq_patterns = [
        r'(?:final answer|answer|choice|option) is\s*:?\s*\(([A-Ea-e])\)',
        r'(?:final answer|answer|choice|option) is\s*:?\s*([A-Ea-e])\b',
        r'\(([A-Ea-e])\)\s*is the correct answer',
        r'\b([A-Ea-e])\)\s*$',
        r'^([A-Ea-e])\)$',
        r'correct label \(character before \'\)\'\)\n([A-Ea-e])',
        r'\b(?:Answer|Choice|Option):\s*([A-Ea-e])\b',
    ]

    if lines:
        last_line = lines[-1].strip()
        for pattern in mcq_patterns:
            match = re.search(pattern, last_line, re.IGNORECASE)
            if match:
                ans = match.group(1).upper()
                logger_extractors.debug(f"Extracted MCQ label '{ans}' from last line")
                return ans
        if re.fullmatch(r'[A-Ea-e]', last_line):
            ans = last_line.upper()
            logger_extractors.debug(f"Extracted MCQ label '{ans}' as last line content")
            return ans

    for pattern in mcq_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            ans = match.group(1).upper()
            logger_extractors.debug(f"Extracted MCQ label '{ans}' from full text")
            return ans

    return None


def _extract_numerical_heuristic(text: str) -> Optional[str]:
    text = str(text).strip()
    lines = text.strip().splitlines()

    gsm_pattern = r'####\s*([+-]?\d+(?:,\d+)*\.?\d*)'
    boxed_pattern = r'\\boxed\{([+-]?\d+(?:,\d+)*\.?\d*)\}'

    match = re.search(gsm_pattern, text)
    if match:
        extracted_num = match.group(1).replace(",", "")
        logger_extractors.debug(f"Extracted numerical answer '{extracted_num}' using GSM pattern")
        return extracted_num

    match = re.search(boxed_pattern, text)
    if match:
        extracted_num = match.group(1).replace(",", "")
        logger_extractors.debug(f"Extracted numerical answer '{extracted_num}' using boxed pattern")
        return extracted_num

    num_pattern_loose = r'[+-]?\d+(?:,\d+)*(?:\.\d+)?'
    numerical_patterns = [
        r'(?:final answer is|the final answer is|final answer:|answer:|the answer is)\s*:?\s*(' + num_pattern_loose + r')\b',
        r'(?:is|equals|result is|=)\s*(' + num_pattern_loose + r')\s*\.?\s*$',
    ]

    if lines:
        last_line = lines[-1].strip()
        for pattern in numerical_patterns:
            match = re.search(pattern, last_line, re.IGNORECASE)
            if match:
                extracted_num = match.group(1).replace(",", "")
                logger_extractors.debug(
                    f"Extracted numerical answer '{extracted_num}' from last line pattern")
                return extracted_num
        if re.fullmatch(num_pattern_loose, last_line.replace("$", "")):
            extracted_num = last_line.replace(",", "").replace("$", "")
            logger_extractors.debug(
                f"Extracted numerical answer '{extracted_num}' as last line content")
            return extracted_num

    for pattern in numerical_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            extracted_num = match.group(1).replace(",", "")
            logger_extractors.debug(
                f"Extracted numerical answer '{extracted_num}' from full text pattern")
            return extracted_num

    numbers = re.findall(num_pattern_loose, text)
    if numbers:
        last_num_str = numbers[-1].replace(",", "")
        logger_extractors.debug(f"Extracted numerical answer '{last_num_str}' as last number found")
        return last_num_str

    return None


def _extract_boolean_heuristic(text: str) -> Optional[str]:
    text = str(text).strip().lower()
    lines = text.strip().splitlines()

    if lines:
        last_line = lines[-1].strip().lower().rstrip(".")
        if last_line == "yes": return "yes"
        if last_line == "no": return "no"
        if last_line == "true": return "yes"
        if last_line == "false": return "no"
        match = re.search(r'\b(?:answer|result) is\s+(yes|no|true|false)\b', last_line)
        if match:
            ans = match.group(1)
            logger_extractors.debug(f"Extracted boolean '{ans}' from last line pattern")
            return "yes" if ans == "true" else "no" if ans == "false" else ans

    match = re.search(r'\b(?:final answer|answer|result) is\s+(yes|no|true|false)\b', text)
    if match:
        ans = match.group(1)
        logger_extractors.debug(f"Extracted boolean '{ans}' from full text pattern")
        return "yes" if ans == "true" else "no" if ans == "false" else ans

    if "yes" in text.split()[-5:]: return "yes"
    if "no" in text.split()[-5:]: return "no"

    return None


def _extract_letter_concat_heuristic(text: str) -> Optional[str]:
    text = str(text).strip()
    lines = text.strip().splitlines()

    pattern = r'(?:final answer|answer|result) is\s*:?\s*([a-zA-Z]+)\b'

    if lines:
        last_line = lines[-1].strip().rstrip(".")
        match = re.search(pattern, last_line, re.IGNORECASE)
        if match:
            ans = match.group(1)
            logger_extractors.debug(f"Extracted letter concat '{ans}' from last line pattern")
            return ans
        if re.fullmatch(r'[a-zA-Z]+', last_line):
            logger_extractors.debug(f"Extracted letter concat '{last_line}' as last line content")
            return last_line

    match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
    if match:
        ans = match.group(1)
        logger_extractors.debug(f"Extracted letter concat '{ans}' from full text pattern")
        return ans

    if lines and re.fullmatch(r'[a-zA-Z]{2,}', lines[-1].strip()):
        logger_extractors.debug(
            f"Extracted letter concat '{lines[-1].strip()}' as fallback last line")
        return lines[-1].strip()

    return None


def extract_answer_heuristic_custom(raw_output: str, dataset_name: str) -> str:
    if not raw_output or str(raw_output).strip() == "[ERROR]" or str(raw_output).strip().startswith("[ERROR:"):
        return "[ERROR]"

    text = str(raw_output).strip()
    extracted: Optional[str] = None

    if dataset_name in ["aqua", "csqa"]:
        extracted = _extract_label_heuristic(text)
    elif dataset_name in ["gsm8k", "multiarith"]:
        extracted = _extract_numerical_heuristic(text)
    elif dataset_name in ["strategyqa", "coin"]:
        extracted = _extract_boolean_heuristic(text)
    elif dataset_name == "letter":
        extracted = _extract_letter_concat_heuristic(text)
    else:
        logger_extractors.warning(
            f"No specific heuristic defined for dataset '{dataset_name}'. Using generic fallback.")
        extracted = _extract_numerical_heuristic(text)
        if extracted is None:
            extracted = _extract_label_heuristic(text)

    if extracted is None:
        logger_extractors.warning(
            f"Could not extract answer for dataset '{dataset_name}' using custom heuristic from: '{text[:150]}...'")
        return "[EXTRACTION_HEURISTIC_FAILURE]"

    return extracted.strip()


MCQ_EXTRACTION_PROMPT = """
Original Question:
{question}

LLM Reasoning and Output:
\"\"\"
{raw_output}
\"\"\"

Analyze the LLM Reasoning and Output based on the Original Question.
The original question is multiple choice with options typically labeled A, B, C, D, E.
Extract ONLY the single capital letter corresponding to the final choice identified in the reasoning or output.
If the reasoning calculates a value, ensure you extract the letter label associated with that value in the options.
If no specific choice label is clearly identified as the final answer, return null.
Return the result as a JSON object with a single key "final_answer" containing the final answer label (A, B, C, D, or E) as a string, or null if not found.

JSON Output:
"""

NUMERICAL_EXTRACTION_PROMPT = """
Original Question:
{question}

LLM Reasoning and Output:
\"\"\"
{raw_output}
\"\"\"

Analyze the LLM Reasoning and Output based on the Original Question.
Extract ONLY the final numerical answer stated in the text. Ignore intermediate calculations.
Look for patterns like "Final Answer: [number]", "#### [number]", or the last number mentioned if it seems conclusive.
Return the result as a JSON object with a single key "final_answer" containing the final numerical answer as a string (e.g., "15", "36.5", "77.5"), or null if no definitive numerical answer is found.

JSON Output:
"""

BOOLEAN_EXTRACTION_PROMPT = """
Original Question:
{question}

LLM Reasoning and Output:
\"\"\"
{raw_output}
\"\"\"

Analyze the LLM Reasoning and Output based on the Original Question.
Extract ONLY the final boolean answer (yes or no) stated in the text.
Look for explicit statements like "Answer: yes", "Final Answer: no", or the concluding yes/no statement.
Return the result as a JSON object with a single key "final_answer" containing either the string "yes" or "no", or null if no definitive boolean answer is found.

JSON Output:
"""

TEXT_EXTRACTION_PROMPT = """
Original Question:
{question}

LLM Reasoning and Output:
\"\"\"
{raw_output}
\"\"\"

Analyze the LLM Reasoning and Output based on the Original Question.
Extract ONLY the final short text answer stated in the text (e.g., a concatenated string of letters for the 'letter' dataset).
Look for patterns like "Answer: [text]" or the concluding text segment if it seems to be the final answer.
Return the result as a JSON object with a single key "final_answer" containing the final text answer as a string, or null if no definitive text answer is found.

JSON Output:
"""


def get_llm_extraction_prompt(question: str, raw_output: str, dataset_name: str) -> str:
    template: str
    if dataset_name in ["aqua", "csqa"]:
        template = MCQ_EXTRACTION_PROMPT
    elif dataset_name in ["gsm8k", "multiarith"]:
        template = NUMERICAL_EXTRACTION_PROMPT
    elif dataset_name in ["strategyqa", "coin"]:
        template = BOOLEAN_EXTRACTION_PROMPT
    elif dataset_name == "letter":
        template = TEXT_EXTRACTION_PROMPT
    else:
        logger_extractors.warning(
            f"No specific LLM prompt template defined for dataset '{dataset_name}'. Using generic numerical fallback.")
        template = NUMERICAL_EXTRACTION_PROMPT

    return template.format(question=question, raw_output=raw_output)
