import pytest

from cogitator.graph_of_thoughts import GraphOfThoughts


def test_run_returns_result_and_calls_prompts(fake_llm_factory, patch_embedding_clustering):
    fake_expansion_config = {"thoughts": ["stepA_sync"]}
    fake_eval_config = {"score": 9, "justification": "Good_sync"}
    llm = fake_llm_factory({
        "generate_sync": fake_expansion_config,
        "json_eval": fake_eval_config,
        "final_answer": "RESULT_sync"
    })
    got_instance = GraphOfThoughts(llm, max_iters=1, num_branches=1, beam_width=1)
    out = got_instance.run("start?")
    assert out == "RESULT_sync"

    expand_call = next(
        (c for c in llm.sync_calls if c["type"] == "generate" and "JSON Steps:" in c["prompt"]),
        None)
    eval_call = next((c for c in llm.sync_calls if
                      c["type"] == "_generate_json_internal" and "JSON Evaluation:" in c["prompt"]),
                     None)
    final_call = next((c for c in llm.sync_calls if
                       c["type"] == "generate" and "Given reasoning steps" in c["prompt"]), None)

    assert expand_call is not None, "Expansion call not found"
    assert eval_call is not None, "Evaluation call not found"
    assert eval_call["response_model"] == "EvaluationResult"
    assert final_call is not None, "Final answer generation call not found"


@pytest.mark.asyncio
async def test_run_async_returns_result_and_calls_prompts(fake_llm_factory,
                                                          patch_embedding_clustering):
    fake_expansion_config_async = {"thoughts": ["stepA_async"]}
    fake_eval_config_async = {"score": 9, "justification": "Good_async"}
    llm = fake_llm_factory({
        "generate_async": fake_expansion_config_async,
        "json_eval": fake_eval_config_async,
        "final_answer": "RESULT_async"
    })
    got_instance = GraphOfThoughts(llm, max_iters=1, num_branches=1, beam_width=1)
    out = await got_instance.run_async("start_async?")
    assert out == "RESULT_async"

    expand_call = next((c for c in llm.async_calls if
                        c["type"] == "generate_async" and "JSON Steps:" in c["prompt"]), None)
    eval_call = next((c for c in llm.async_calls if
                      c["type"] == "_generate_json_internal_async" and "JSON Evaluation:" in c[
                          "prompt"]), None)
    final_call = next((c for c in llm.async_calls if
                       c["type"] == "generate_async" and "Given reasoning steps" in c["prompt"]),
                      None)

    assert expand_call is not None, "Async expansion call not found"
    assert eval_call is not None, "Async evaluation call not found"
    assert eval_call["response_model"] == "EvaluationResult"
    assert final_call is not None, "Async final answer generation call not found"
