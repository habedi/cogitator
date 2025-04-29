import json

import pytest

from cogitator.graph_of_thoughts import GraphOfThoughts


# No BaseLLM or FakeLLM needed here


# Use fixtures from conftest.py
def test_run_returns_result_and_calls_prompts(fake_llm_factory, patch_embedding_clustering):
    llm = fake_llm_factory({
        "json_steps": ["stepA_sync"],
        "json_eval": {"score": 9, "justification": "Good_sync"},
        "final_answer": "RESULT_sync"
    })
    got_instance = GraphOfThoughts(llm, max_iters=1, num_branches=1, beam_width=1)
    out = got_instance.run("start?")
    assert out == "RESULT_sync"
    # Check calls made via generate_json and generate
    assert any("JSON Steps:" in c["prompt"] for c in llm.sync_calls if
               c["type"] == "generate")  # Expansion uses generate then parse
    assert any("JSON Evaluation:" in c["prompt"] for c in llm.sync_calls if c["type"] == "json")
    assert any(
        "Given reasoning steps" in c["prompt"] for c in llm.sync_calls if c["type"] == "generate")


@pytest.mark.asyncio
async def test_run_async_returns_result_and_calls_prompts(fake_llm_factory,
                                                          patch_embedding_clustering):
    llm = fake_llm_factory({
        "generate_async": json.dumps(["stepA_async"]),  # Expansion now uses generate_async
        "json_eval": {"score": 9, "justification": "Good_async"},  # Eval uses generate_json_async
        "final_answer": "RESULT_async"  # Final answer via generate_async
    })
    got_instance = GraphOfThoughts(llm, max_iters=1, num_branches=1, beam_width=1)
    out = await got_instance.run_async("start_async?")
    assert out == "RESULT_async"
    # Check calls made via generate_async and generate_json_async
    assert any(
        "JSON Steps:" in c["prompt"] for c in llm.async_calls if c["type"] == "generate_async")
    assert any(
        "JSON Evaluation:" in c["prompt"] for c in llm.async_calls if c["type"] == "async_json")
    assert any("Given reasoning steps" in c["prompt"] for c in llm.async_calls if
               c["type"] == "generate_async")
