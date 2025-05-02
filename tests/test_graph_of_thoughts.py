# tests/test_graph_of_thoughts.py
import pytest
import json # Import json

from cogitator.graph_of_thoughts import GraphOfThoughts
from cogitator.schemas import EvaluationResult # Import schema


def test_run_returns_result_and_calls_prompts(fake_llm_factory, patch_embedding_clustering):
    # Provide JSON string for expansion, configure under generate_sync key
    fake_expansion_config_str = json.dumps({"thoughts": ["stepA_sync"]})
    fake_eval_config = EvaluationResult(score=9, justification="Good_sync")
    llm = fake_llm_factory({
        "generate_sync": fake_expansion_config_str, # GoT uses generate for expansion if use_json=False
        "json_eval": fake_eval_config,
        "final_answer": "RESULT_sync"
    })
    # Instantiate GoT with use_json=False so it calls llm.generate for expansion
    got_instance = GraphOfThoughts(llm, max_iters=1, num_branches=1, beam_width=1, use_json=False)
    out = got_instance.run("start?")
    assert out == "RESULT_sync"

    # Check the correct call types
    expand_call = next(
        (c for c in llm.sync_calls if c["type"] == "generate" and "JSON list of strings" in c["prompt"]),
        None)
    eval_call = next((c for c in llm.sync_calls if
                      c["type"] == "_generate_json_internal" and "JSON Evaluation:" in c["prompt"]),
                     None)
    final_call = next((c for c in llm.sync_calls if
                       c["type"] == "generate" and ("Given reasoning steps" in c["prompt"] or c["prompt"].startswith("Answer the question:"))),
                      None)

    assert expand_call is not None, "Expansion call not found"
    assert eval_call is not None, "Evaluation call not found"
    assert eval_call["response_model"] == "EvaluationResult"
    assert final_call is not None, "Final answer generation call not found"


@pytest.mark.asyncio
async def test_run_async_returns_result_and_calls_prompts(fake_llm_factory,
                                                          patch_embedding_clustering):
    # Provide JSON string for expansion, configure under generate_async key
    fake_expansion_config_async_str = json.dumps({"thoughts": ["stepA_async"]})
    fake_eval_config_async = EvaluationResult(score=9, justification="Good_async")
    llm = fake_llm_factory({
        "generate_async": fake_expansion_config_async_str, # GoT uses generate_async if use_json=False
        "json_eval": fake_eval_config_async,
        "final_answer": "RESULT_async"
    })
    # Instantiate GoT with use_json=False so it calls llm.generate_async for expansion
    got_instance = GraphOfThoughts(llm, max_iters=1, num_branches=1, beam_width=1, use_json=False)
    out = await got_instance.run_async("start_async?")
    assert out == "RESULT_async"

    # Check the correct call types
    expand_call = next((c for c in llm.async_calls if
                        c["type"] == "generate_async" and "JSON list of strings" in c["prompt"]),
                       None)
    eval_call = next((c for c in llm.async_calls if
                      c["type"] == "_generate_json_internal_async" and "JSON Evaluation:" in c["prompt"]),
                     None)
    final_call = next((c for c in llm.async_calls if
                       c["type"] == "generate_async" and ("Given reasoning steps" in c["prompt"] or c["prompt"].startswith("Answer the question:"))),
                      None)

    assert expand_call is not None, "Async expansion call not found"
    assert eval_call is not None, "Async evaluation call not found"
    assert eval_call["response_model"] == "EvaluationResult"
    assert final_call is not None, "Async final answer generation call not found"
