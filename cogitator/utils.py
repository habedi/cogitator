"""Provides utility functions used across the Cogitator library."""

import re


def count_steps(cot: str) -> int:
    """Counts the number of reasoning steps in a Chain-of-Thought string.

    Identifies steps based on lines starting with digits followed by a period/paren
    or lines starting with list markers like -, *, •.

    Args:
        cot: The string containing the Chain-of-Thought reasoning.

    Returns:
        The integer count of identified reasoning steps.
    """
    return sum(1 for line in cot.splitlines() if re.match(r"^(\d+[\.\)]|[-*•])\s+", line.strip()))


def approx_token_length(text: str) -> int:
    """Approximates the number of tokens in a string.

    Counts sequences of word characters and any non-whitespace, non-word characters
    as separate tokens. This provides a rough estimate, not a precise token count
    based on a specific tokenizer model.

    Args:
        text: The input string.

    Returns:
        An approximate integer count of tokens.
    """
    return len(re.findall(r"\w+|[^\w\s]", text))


def exact_match(pred: str, gold: str) -> bool:
    """Performs case-insensitive exact matching between two strings.

    Strips leading/trailing whitespace and converts both strings to lowercase
    before comparison.

    Args:
        pred: The predicted string.
        gold: The ground truth (gold standard) string.

    Returns:
        True if the normalized strings are identical, False otherwise.
    """
    return pred.strip().lower() == gold.strip().lower()


def accuracy(preds: list[str], golds: list[str]) -> float:
    """Calculates the exact match accuracy between lists of predictions and golds.

    Uses the `exact_match` function for comparison. Handles potential differences
    in list lengths by iterating up to the length of the shorter list if `strict=False`
    (default in zip). If `golds` is empty, returns 0.0.

    Args:
        preds: A list of predicted strings.
        golds: A list of ground truth strings.

    Returns:
        The accuracy score as a float between 0.0 and 1.0.
    """
    if not golds:
        return 0.0
    matches = sum(exact_match(p, g) for p, g in zip(preds, golds, strict=False))
    return matches / len(golds)
