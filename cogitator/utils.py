import re


def count_steps(cot: str) -> int:
    return sum(1 for line in cot.splitlines() if re.match(r"^(\d+[\.\)]|[-*•])\s+", line.strip()))


def approx_token_length(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text))


def exact_match(pred: str, gold: str) -> bool:
    return pred.strip().lower() == gold.strip().lower()


def accuracy(preds: list[str], golds: list[str]) -> float:
    if not golds:
        return 0.0
    matches = sum(exact_match(p, g) for p, g in zip(preds, golds, strict=False))
    return matches / len(golds)
