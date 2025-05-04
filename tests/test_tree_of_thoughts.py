import pytest

from cogitator import TreeOfThoughts
from cogitator.schemas import ThoughtExpansion, EvaluationResult


def test_run_returns_final_and_calls_prompts(fake_llm_factory):
    fake_expansion = ThoughtExpansion(thoughts=["step1_sync"])
    fake_eval = EvaluationResult(score=8, justification="Okay_sync")
    llm = fake_llm_factory({
        "json_steps": fake_expansion,
        "json_eval": fake_eval,
        "final_answer": "FINAL_sync"
    })
    tot = TreeOfThoughts(llm, max_depth=1, num_branches=1, sims=1, c_puct=1.0)
    out = tot.run("test?")

    assert out == "FINAL_sync"

    expand_call = next((c for c in llm.sync_calls if
                        c["type"] == "_generate_json_internal" and "JSON Output:" in c[
                            "prompt"] and "thoughts" in c["prompt"]),
                       None)
    eval_call = next((c for c in llm.sync_calls if
                      c["type"] == "_generate_json_internal" and "JSON Evaluation:" in c["prompt"]),
                     None)
    final_call = next((c for c in llm.sync_calls if
                       c["type"] == "generate" and (
                           "Given reasoning steps" in c["prompt"] or c["prompt"].startswith(
                           "Answer the question:"))),
                      None)

    assert expand_call is not None, "Expansion call not found"
    assert expand_call["response_model"] == "ThoughtExpansion"
    assert eval_call is not None, "Evaluation call not found"
    assert eval_call["response_model"] == "EvaluationResult"
    assert final_call is not None, "Final answer generation call not found"


@pytest.mark.asyncio
async def test_run_async_returns_final_and_calls_prompts(fake_llm_factory):
    fake_expansion_async = ThoughtExpansion(thoughts=["step1_async"])
    fake_eval_async = EvaluationResult(score=8, justification="Okay_async")
    llm = fake_llm_factory({
        "json_steps": fake_expansion_async,
        "json_eval": fake_eval_async,
        "final_answer": "FINAL_async"
    })
    tot = TreeOfThoughts(llm, max_depth=1, num_branches=1, sims=1, c_puct=1.0)
    out = await tot.run_async("test_async?")

    assert out == "FINAL_async"

    expand_call = next((c for c in llm.async_calls if
                        c["type"] == "_generate_json_internal_async" and "JSON Output:" in c[
                            "prompt"] and "thoughts" in c["prompt"]),
                       None)
    eval_call = next((c for c in llm.async_calls if
                      c["type"] == "_generate_json_internal_async" and "JSON Evaluation:" in c[
                          "prompt"]),
                     None)
    final_call = next((c for c in llm.async_calls if
                       c["type"] == "generate_async" and (
                           "Given reasoning steps" in c["prompt"] or c["prompt"].startswith(
                           "Answer the question:"))),
                      None)

    assert expand_call is not None, "Async expansion call not found"
    assert expand_call["response_model"] == "ThoughtExpansion"
    assert eval_call is not None, "Async evaluation call not found"
    assert eval_call["response_model"] == "EvaluationResult"
    assert final_call is not None, "Async final answer generation call not found"
