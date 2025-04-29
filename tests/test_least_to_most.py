import pytest
import json # Import json for the invalid JSON test cases

from cogitator.least_to_most import LeastToMost



def test_decompose_numbered(fake_llm_factory):
    llm = fake_llm_factory({"json_subquestions": ["sub1", "sub2", "sub3"]})
    ltm = LeastToMost(llm)
    subs = ltm.decompose("anything")
    assert subs == ["sub1", "sub2", "sub3"]


def test_decompose_bullets(fake_llm_factory):
    llm = fake_llm_factory({"json_subquestions": ["subA", "subB"]})
    ltm = LeastToMost(llm)
    subs = ltm.decompose("anything")
    assert subs == ["subA", "subB"]


def test_decompose_fallback_simulated(fake_llm_factory):
    llm = fake_llm_factory({"json_subquestions": ["This is a sentence", "Another sentence"]})
    ltm = LeastToMost(llm)
    subs = ltm.decompose("anything")
    assert subs == ["This is a sentence", "Another sentence"]


def test_decompose_empty_list_raises(fake_llm_factory):
    llm = fake_llm_factory({"json_subquestions": []})
    ltm = LeastToMost(llm)
    # Match the exact error message raised when the subquestion list is empty after parsing
    with pytest.raises(ValueError, match="LLM returned empty or invalid subquestions list after parsing."):
        ltm.decompose("anything")


def test_decompose_invalid_json_raises(fake_llm_factory):
    llm = fake_llm_factory()
    ltm = LeastToMost(llm)
    # Construct the expected prompt key used by the fake LLM
    prompt_key = ltm._build_prefix() + ltm.decompose_prompt_template.format(question="anything")
    llm.responses_map[prompt_key] = "this is not json"
    # Match the error message raised when the underlying JSON call fails
    with pytest.raises(ValueError, match="Failed to decompose question due to LLM error:"):
        ltm.decompose("anything")


def test_max_subqs_trims(fake_llm_factory):
    many_subs = [f"sub{i + 1}" for i in range(20)]
    llm = fake_llm_factory({"json_subquestions": many_subs})
    ltm = LeastToMost(llm, max_subqs=5)
    subs = ltm.decompose("anything")
    assert len(subs) == 5
    assert subs == ["sub1", "sub2", "sub3", "sub4", "sub5"]


def test_answer_integration(fake_llm_factory):
    llm = fake_llm_factory({
        "json_subquestions": ["dummy sub1", "dummy sub2"],
        "sub_answer": "sub_answer_sync",
        "final_answer": "ANS_sync"
    })
    ltm = LeastToMost(llm)
    out = ltm("question?")
    # Check the expected final answer based on fake LLM config
    assert out == "ANS_sync"



@pytest.mark.asyncio
async def test_decompose_async_numbered(fake_llm_factory):
    llm = fake_llm_factory({"json_subquestions": ["sub1", "sub2", "sub3"]})
    ltm = LeastToMost(llm)
    subs = await ltm.decompose_async("anything async")
    assert subs == ["sub1", "sub2", "sub3"]


@pytest.mark.asyncio
async def test_decompose_async_empty_list_raises(fake_llm_factory):
    llm = fake_llm_factory({"json_subquestions": []})
    ltm = LeastToMost(llm)
    # Match the exact error message raised when the async subquestion list is empty
    with pytest.raises(ValueError, match="Async LLM returned empty or invalid subquestions list after parsing."):
        await ltm.decompose_async("anything async")


@pytest.mark.asyncio
async def test_decompose_async_invalid_json_raises(fake_llm_factory):
    llm = fake_llm_factory()
    ltm = LeastToMost(llm)
    # Construct the expected prompt key used by the fake LLM
    prompt_key = ltm._build_prefix() + ltm.decompose_prompt_template.format(question="anything async")
    llm.responses_map[prompt_key] = "this is not json async"
    # Match the error message raised when the underlying async JSON call fails
    with pytest.raises(ValueError, match="Async decomposition failed due to LLM error:"):
        await ltm.decompose_async("anything async")


@pytest.mark.asyncio
async def test_answer_async_integration(fake_llm_factory):
    # Configure fake LLM for async path
    llm = fake_llm_factory({
        "json_subquestions": ["dummy sub1 async", "dummy sub2 async"],
        "sub_answer": "sub_answer_async_val", # Specific value for sub answer
        "final_answer": "ANS_async_val" # Specific value for final answer
    })
    ltm = LeastToMost(llm)
    out = await ltm.answer_async("question async?")
    # Check the expected final answer based on fake LLM config
    assert out == "ANS_async_val"


@pytest.mark.asyncio
async def test_solve_async_calls_generate_async(fake_llm_factory):
    # Configure a specific async response for sub_answer
    expected_sub_answer = "async_sub_answer_test"
    llm = fake_llm_factory({"sub_answer": expected_sub_answer})
    ltm = LeastToMost(llm)
    solved = await ltm.solve_async("main q", ["sub1", "sub2"])
    # Check if the answers match the configured async sub_answer
    assert solved == [("sub1", expected_sub_answer), ("sub2", expected_sub_answer)]

