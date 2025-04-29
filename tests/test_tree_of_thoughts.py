import pytest

from cogitator.tree_of_thoughts import TreeOfThoughts


def test_run_returns_final_and_calls_prompts(fake_llm_factory):
    llm = fake_llm_factory({
        "json_steps": ["step1_sync"],
        "json_eval": {"score": 8, "justification": "Okay_sync"},
        "final_answer": "FINAL_sync"
    })
    tot = TreeOfThoughts(llm, max_depth=1, num_branches=1, sims=1, c_puct=1.0)
    out = tot.run("test?")
    assert out == "FINAL_sync"
    assert any("JSON Steps:" in c["prompt"] for c in llm.sync_calls if c["type"] == "json")
    assert any("JSON Evaluation:" in c["prompt"] for c in llm.sync_calls if c["type"] == "json")
    assert any(
        "Given reasoning steps" in c["prompt"] for c in llm.sync_calls if c["type"] == "generate")


@pytest.mark.asyncio
async def test_run_async_returns_final_and_calls_prompts(fake_llm_factory):
    llm = fake_llm_factory({
        "json_steps": ["step1_async"],
        "json_eval": {"score": 8, "justification": "Okay_async"},
        "final_answer": "FINAL_async"
    })
    tot = TreeOfThoughts(llm, max_depth=1, num_branches=1, sims=1, c_puct=1.0)
    out = await tot.run_async("test_async?")
    assert out == "FINAL_async"
    assert any("JSON Steps:" in c["prompt"] for c in llm.async_calls if c["type"] == "async_json")
    assert any(
        "JSON Evaluation:" in c["prompt"] for c in llm.async_calls if c["type"] == "async_json")
    assert any("Given reasoning steps" in c["prompt"] for c in llm.async_calls if
               c["type"] == "generate_async")
