import pytest

from cogitator.tree_of_thoughts import TreeOfThoughts


# Import schemas used by TreeOfThoughts


def test_run_returns_final_and_calls_prompts(fake_llm_factory):
    # Config needs to provide data that fake _generate_json_internal will return as string
    fake_expansion = {"thoughts": ["step1_sync"]}
    fake_eval = {"score": 8, "justification": "Okay_sync"}
    llm = fake_llm_factory({
        # Configure responses based on prompt keywords used by _get_response_for_prompt
        "json_steps": fake_expansion,
        "json_eval": fake_eval,
        "final_answer": "FINAL_sync"
    })
    tot = TreeOfThoughts(llm, max_depth=1, num_branches=1, sims=1, c_puct=1.0)
    out = tot.run("test?")

    assert out == "FINAL_sync"

    # Verify internal JSON generation calls were made with correct models
    expand_call = next((c for c in llm.sync_calls if
                        c["type"] == "_generate_json_internal" and "JSON Steps:" in c["prompt"]),
                       None)
    eval_call = next((c for c in llm.sync_calls if
                      c["type"] == "_generate_json_internal" and "JSON Evaluation:" in c["prompt"]),
                     None)
    final_call = next((c for c in llm.sync_calls if
                       c["type"] == "generate" and "Given reasoning steps" in c["prompt"]), None)

    assert expand_call is not None, "Expansion call not found"
    assert expand_call["response_model"] == "ThoughtExpansion"

    assert eval_call is not None, "Evaluation call not found"
    assert eval_call["response_model"] == "EvaluationResult"

    assert final_call is not None, "Final answer generation call not found"


@pytest.mark.asyncio
async def test_run_async_returns_final_and_calls_prompts(fake_llm_factory):
    # Config needs to provide data that fake _generate_json_internal_async will return as string
    fake_expansion_async = {"thoughts": ["step1_async"]}
    fake_eval_async = {"score": 8, "justification": "Okay_async"}
    llm = fake_llm_factory({
        # Configure responses based on prompt keywords used by _get_response_for_prompt
        "json_steps": fake_expansion_async,  # Will be picked by keyword match
        "json_eval": fake_eval_async,  # Will be picked by keyword match
        "final_answer": "FINAL_async"  # Will be picked by keyword match
    })
    tot = TreeOfThoughts(llm, max_depth=1, num_branches=1, sims=1, c_puct=1.0)
    out = await tot.run_async("test_async?")

    assert out == "FINAL_async"

    # Verify internal async JSON generation calls were made with correct models
    expand_call = next((c for c in llm.async_calls if
                        c["type"] == "_generate_json_internal_async" and "JSON Steps:" in c[
                            "prompt"]), None)
    eval_call = next((c for c in llm.async_calls if
                      c["type"] == "_generate_json_internal_async" and "JSON Evaluation:" in c[
                          "prompt"]), None)
    final_call = next((c for c in llm.async_calls if
                       c["type"] == "generate_async" and "Given reasoning steps" in c["prompt"]),
                      None)

    assert expand_call is not None, "Async expansion call not found"
    assert expand_call["response_model"] == "ThoughtExpansion"

    assert eval_call is not None, "Async evaluation call not found"
    assert eval_call["response_model"] == "EvaluationResult"

    assert final_call is not None, "Async final answer generation call not found"
