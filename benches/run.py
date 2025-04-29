# benches/run.py
import os
import time

import polars as pl
from cogitator.auto_cot import AutoCoT
from cogitator.cdw_cot import CDWCoT
from cogitator.graph_of_thoughts import GraphOfThoughts
from cogitator.least_to_most import LeastToMost
from cogitator.model import OpenAILLM
from cogitator.sc_cot import SelfConsistency
from cogitator.tree_of_thoughts import TreeOfThoughts
from cogitator.utils import accuracy
from datasets import load_dataset


class Datasets:
    @staticmethod
    def load_gsm8k():
        ds = load_dataset("openai/gsm8k", "main", split="test")
        return ds["question"], ds["answer"]

    @staticmethod
    def load_multiarith():
        ds = load_dataset("ChilleD/MultiArith", split="train")
        return ds["problem"], ds["solution"]

    @staticmethod
    def load_aqua_rat():
        ds = load_dataset("deepmind/aqua_rat", split="test")
        return ds["question"], ds["answer"]

    @staticmethod
    def load_commonsense_qa():
        ds = load_dataset("tau/commonsense_qa", split="train")
        qs, golds = [], []
        for item in ds:
            qs.append(item["question"])
            idx = ord(item["answerKey"]) - ord("A")
            golds.append(item["choices"]["text"][idx])
        return qs, golds

    @staticmethod
    def load_strategy_qa():
        ds = load_dataset("voidful/StrategyQA", split="train")
        return ds["question"], ds["answer"]

    @staticmethod
    def load_coin_flip():
        ds = load_dataset("skrishna/coin_flip", split="train")
        return ds["question"], ds["answer"]

    @staticmethod
    def load_last_letter():
        ds = load_dataset("ChilleD/LastLetterConcat", split="train")
        cols = ds.column_names
        if "question" in cols and "answer" in cols:
            return ds["question"], ds["answer"]
        if "input" in cols and "output" in cols:
            return ds["input"], ds["output"]
        return ds[cols[0]], ds[cols[1]]


def run_benchmark(name, model, questions, golds):
    preds, times = [], []
    for q in questions:
        t0 = time.time()
        out = model(q)
        times.append(time.time() - t0)
        preds.append(out)
    return {
        "method": name,
        "accuracy": accuracy(preds, golds),
        "avg_time_s": sum(times) / len(times),
        "num_queries": len(questions),
    }


def main():
    questions, golds = Datasets.load_gsm8k()
    llm = OpenAILLM(api_key=os.getenv("OPENAI_API_KEY"))

    auto = AutoCoT(llm)
    auto.fit(questions)
    cdw = CDWCoT(llm)
    cdw.init_pool(questions, golds)
    cdw.train()
    scot = SelfConsistency(llm, n_samples=10)
    ltm = LeastToMost(llm)
    tot = TreeOfThoughts(llm)
    got = GraphOfThoughts(llm)

    methods = [
        ("Zero‐Shot‐CoT", lambda q: llm.generate(f"Q: {q}\nA: Let's think step by step.")),
        ("AutoCoT", auto),
        ("CDWCoT", cdw),
        ("SelfConsistency", scot),
        ("LeastToMost", ltm),
        ("TreeOfThoughts", tot),
        ("GraphOfThoughts", got),
        (
            "AutoCoT+SelfConsistency",
            lambda q: scot.run(
                "\n\n".join(auto.demos) + f"\n\nQ: {q}\nA: Let's think step by step."
            ),
        ),
    ]

    results = [run_benchmark(name, model, questions, golds) for name, model in methods]
    df = pl.DataFrame(results)
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
