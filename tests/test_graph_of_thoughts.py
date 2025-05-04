import json

import pytest

from cogitator import EvaluationResult, ExtractedAnswer
from cogitator import GraphOfThoughts


def test_run_returns_result_and_calls_prompts_text_format(
    fake_llm_factory, patch_embedding_clustering):
    fake_expansion_config_str = json.dumps({"thoughts": ["stepA_sync"]})
    fake_eval_config = EvaluationResult(score=9, justification="Good_sync")
    llm = fake_llm_factory({
        "generate_sync": fake_expansion_config_str,
        "json_eval": fake_eval_config,
        "final_answer": "RESULT_sync_text"
    })

    got_instance = GraphOfThoughts(llm, max_iters=1, num_branches=1, beam_width=1,
                                   final_answer_format="text")
    out = got_instance.run("start?")
    assert out == "RESULT_sync_text"

    expand_call = next(
        (c for c in llm.sync_calls if
         c["type"] == "generate" and "JSON list of strings" in c["prompt"]),
        None)
    eval_call = next((c for c in llm.sync_calls if
                      c["type"] == "_generate_json_internal" and "JSON Evaluation:" in c["prompt"]),
                     None)
    final_call = next((c for c in llm.sync_calls if
                       c["type"] == "generate" and c["prompt"].startswith("Given reasoning steps")),
                      None)

    assert expand_call is not None, "Expansion call not found"
    assert eval_call is not None, "Evaluation call not found"
    assert eval_call["response_model"] == "EvaluationResult"
    assert final_call is not None, "Final answer generation call (text) not found"


def test_run_returns_result_and_calls_prompts_json_format(
    fake_llm_factory, patch_embedding_clustering):
    fake_expansion_config_str = json.dumps({"thoughts": ["stepA_sync_json"]})
    fake_eval_config = EvaluationResult(score=9, justification="Good_sync_json")
    fake_final_answer_obj = ExtractedAnswer(final_answer="RESULT_sync_json")
    llm = fake_llm_factory({
        "generate_sync": fake_expansion_config_str,
        "json_eval": fake_eval_config,
        "json_answer": fake_final_answer_obj
    })

    got_instance = GraphOfThoughts(llm, max_iters=1, num_branches=1, beam_width=1,
                                   final_answer_format="json")
    out = got_instance.run("start_json?")
    assert out == "RESULT_sync_json"

    final_json_call = next((c for c in llm.sync_calls if
                            c["type"] == "_generate_json_internal" and "JSON Answer:" in c[
                                "prompt"]),
                           None)

    assert final_json_call is not None, "Final answer generation call (JSON) not found"
    assert final_json_call["response_model"] == "ExtractedAnswer"


@pytest.mark.asyncio
async def test_run_async_returns_result_and_calls_prompts_text_format(
    fake_llm_factory, patch_embedding_clustering):
    fake_expansion_config_async_str = json.dumps({"thoughts": ["stepA_async"]})
    fake_eval_config_async = EvaluationResult(score=9, justification="Good_async")
    llm = fake_llm_factory({
        "generate_async": fake_expansion_config_async_str,
        "json_eval": fake_eval_config_async,
        "final_answer": "RESULT_async_text"
    })

    got_instance = GraphOfThoughts(llm, max_iters=1, num_branches=1, beam_width=1,
                                   final_answer_format="text")
    out = await got_instance.run_async("start_async?")
    assert out == "RESULT_async_text"

    expand_call = next((c for c in llm.async_calls if
                        c["type"] == "generate_async" and "JSON list of strings" in c["prompt"]),
                       None)
    eval_call = next((c for c in llm.async_calls if
                      c["type"] == "_generate_json_internal_async" and "JSON Evaluation:" in c[
                          "prompt"]),
                     None)
    final_call = next((c for c in llm.async_calls if
                       c["type"] == "generate_async" and c["prompt"].startswith(
                           "Given reasoning steps")),
                      None)

    assert expand_call is not None, "Async expansion call not found"
    assert eval_call is not None, "Async evaluation call not found"
    assert eval_call["response_model"] == "EvaluationResult"
    assert final_call is not None, "Async final answer generation call (text) not found"


@pytest.mark.asyncio
async def test_run_async_returns_result_and_calls_prompts_json_format(
    fake_llm_factory, patch_embedding_clustering):
    fake_expansion_config_async_str = json.dumps({"thoughts": ["stepA_async_json"]})
    fake_eval_config_async = EvaluationResult(score=9, justification="Good_async_json")
    fake_final_answer_obj_async = ExtractedAnswer(final_answer="RESULT_async_json")
    llm = fake_llm_factory({
        "generate_async": fake_expansion_config_async_str,
        "json_eval": fake_eval_config_async,
        "json_answer": fake_final_answer_obj_async
    })

    got_instance = GraphOfThoughts(llm, max_iters=1, num_branches=1, beam_width=1,
                                   final_answer_format="json")
    out = await got_instance.run_async("start_async_json?")
    assert out == "RESULT_async_json"

    final_json_call = next((c for c in llm.async_calls if
                            c["type"] == "_generate_json_internal_async" and "JSON Answer:" in c[
                                "prompt"]),
                           None)

    assert final_json_call is not None, "Async final answer generation call (JSON) not found"
    assert final_json_call["response_model"] == "ExtractedAnswer"
