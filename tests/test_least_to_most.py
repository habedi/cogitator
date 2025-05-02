# tests/test_least_to_most.py

import json
import asyncio
import pytest

from pydantic import ValidationError

from cogitator.least_to_most import LeastToMost
from cogitator.schemas import LTMDecomposition, ExtractedAnswer


def test_decompose_numbered(fake_llm_factory):
    llm = fake_llm_factory({
        "json_subquestions": LTMDecomposition(subquestions=["sub1", "sub2", "sub3"])
    })
    # Instantiate with use_json_parsing=True to match default behavior if not specified
    ltm = LeastToMost(llm, use_json_parsing=True)
    subs = ltm.decompose("anything")
    assert subs == ["sub1", "sub2", "sub3"]

    assert len(llm.sync_calls) == 1
    assert llm.sync_calls[0]["type"] == "_generate_json_internal"
    assert llm.sync_calls[0]["response_model"] == "LTMDecomposition"


def test_decompose_bullets(fake_llm_factory):
    llm = fake_llm_factory({
        "json_subquestions": LTMDecomposition(subquestions=["subA", "subB"])
    })
    ltm = LeastToMost(llm, use_json_parsing=True)
    subs = ltm.decompose("anything")
    assert subs == ["subA", "subB"]


def test_decompose_fallback_simulated(fake_llm_factory):
    llm = fake_llm_factory({
        "json_subquestions": LTMDecomposition(
            subquestions=["This is a sentence", "Another sentence"])
    })
    ltm = LeastToMost(llm, use_json_parsing=True)
    subs = ltm.decompose("anything")
    assert subs == ["This is a sentence", "Another sentence"]


def test_decompose_empty_list_raises(fake_llm_factory):
    llm = fake_llm_factory({
        "json_subquestions": LTMDecomposition(subquestions=[])
    })
    ltm = LeastToMost(llm, use_json_parsing=True)

    with pytest.raises(ValueError, match="LLM returned empty subquestions list after validation."):
        ltm.decompose("anything")


def test_decompose_invalid_json_raises(fake_llm_factory):
    llm = fake_llm_factory()
    # Simulate the RuntimeError that generate_json would raise after catching JSONDecodeError
    llm.generate_json = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("generate_json failed... Last error: JSONDecodeError..."))

    ltm = LeastToMost(llm, use_json_parsing=True)

    # --- FIX: Expect "LLM error: RuntimeError" ---
    with pytest.raises(ValueError, match=r"Failed to decompose question due to LLM error: RuntimeError"):
        ltm.decompose("anything")


def test_decompose_invalid_schema_raises(fake_llm_factory):
    llm = fake_llm_factory()
    # Simulate the RuntimeError that generate_json would raise after catching ValidationError
    llm.generate_json = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("generate_json failed... Last error: ValidationError..."))

    ltm = LeastToMost(llm, use_json_parsing=True)

    # --- FIX: Expect "LLM error: RuntimeError" ---
    with pytest.raises(ValueError, match=r"Failed to decompose question due to LLM error: RuntimeError"):
        ltm.decompose("anything")


def test_max_subqs_trims(fake_llm_factory):
    many_subs = [f"sub{i + 1}" for i in range(20)]
    llm = fake_llm_factory({
        "json_subquestions": LTMDecomposition(subquestions=many_subs)
    })
    ltm = LeastToMost(llm, max_subqs=5, use_json_parsing=True)
    subs = ltm.decompose("anything")
    assert len(subs) == 5
    assert subs == ["sub1", "sub2", "sub3", "sub4", "sub5"]


def test_answer_integration(fake_llm_factory):
    llm = fake_llm_factory({
        "json_subquestions": LTMDecomposition(subquestions=["dummy sub1", "dummy sub2"]),
        "sub_answer": "sub_answer_sync",
        "final_answer": "ANS_sync" # This key should now be matched by the fake LLM
    })
    # Make LtM use non-JSON mode for solve/final steps
    ltm = LeastToMost(llm, use_json_parsing=False)
    out = ltm("question?")
    assert out == "ANS_sync" # Assertion should pass now

    # Check if the correct types of calls were made
    assert any(c["type"] == "_generate_json_internal" and c["response_model"] == "LTMDecomposition" for c in llm.sync_calls)
    assert any(c["type"] == "generate" and "Current Subquestion:" in c["prompt"] for c in llm.sync_calls)
    # Check the specific LtM final prompt structure
    assert any(c["type"] == "generate" and c["prompt"].startswith("Based on the following") and "Final Answer:" in c["prompt"] for c in llm.sync_calls)


@pytest.mark.asyncio
async def test_decompose_async_numbered(fake_llm_factory):
    llm = fake_llm_factory({
        "json_subquestions": LTMDecomposition(subquestions=["sub1", "sub2", "sub3"])
    })
    ltm = LeastToMost(llm, use_json_parsing=True)
    subs = await ltm.decompose_async("anything async")
    assert subs == ["sub1", "sub2", "sub3"]

    assert len(llm.async_calls) == 1
    assert llm.async_calls[0]["type"] == "_generate_json_internal_async"
    assert llm.async_calls[0]["response_model"] == "LTMDecomposition"


@pytest.mark.asyncio
async def test_decompose_async_empty_list_raises(fake_llm_factory):
    llm = fake_llm_factory({
        "json_subquestions": LTMDecomposition(subquestions=[])
    })
    ltm = LeastToMost(llm, use_json_parsing=True)

    with pytest.raises(ValueError,
                       match="Async LLM returned empty subquestions list after validation."):
        await ltm.decompose_async("anything async")


@pytest.mark.asyncio
async def test_decompose_async_invalid_json_raises(fake_llm_factory):
    llm = fake_llm_factory()
    async def mock_fail(*args, **kwargs):
        # Simulate the RuntimeError that generate_json_async would raise
        raise RuntimeError("generate_json_async failed... Last error: JSONDecodeError...")
    llm.generate_json_async = mock_fail

    ltm = LeastToMost(llm, use_json_parsing=True)

    # --- FIX: Expect "LLM error: RuntimeError" ---
    with pytest.raises(ValueError, match=r"Async decomposition failed due to LLM error: RuntimeError"):
        await ltm.decompose_async("anything async")


@pytest.mark.asyncio
async def test_decompose_async_invalid_schema_raises(fake_llm_factory):
    llm = fake_llm_factory()
    async def mock_fail(*args, **kwargs):
        # Simulate the RuntimeError that generate_json_async would raise
        raise RuntimeError("generate_json_async failed... Last error: ValidationError...")
    llm.generate_json_async = mock_fail

    ltm = LeastToMost(llm, use_json_parsing=True)

    # --- FIX: Expect "LLM error: RuntimeError" ---
    with pytest.raises(ValueError, match=r"Async decomposition failed due to LLM error: RuntimeError"):
        await ltm.decompose_async("anything async")


@pytest.mark.asyncio
async def test_answer_async_integration(fake_llm_factory):
    llm = fake_llm_factory({
        "json_subquestions": LTMDecomposition(
            subquestions=["dummy sub1 async", "dummy sub2 async"]),
        "sub_answer": "sub_answer_async_val",
        "final_answer": "ANS_async_val" # This key should now be matched
    })
    ltm = LeastToMost(llm, use_json_parsing=False)
    out = await ltm.answer_async("question async?")
    assert out == "ANS_async_val" # Assertion should pass now

    assert any(c["type"] == "_generate_json_internal_async" and c["response_model"] == "LTMDecomposition" for c in llm.async_calls)
    assert any(c["type"] == "generate_async" and "Current Subquestion:" in c["prompt"] for c in llm.async_calls)
    # Check the specific LtM final prompt structure
    assert any(c["type"] == "generate_async" and c["prompt"].startswith("Based on the following") and "Final Answer:" in c["prompt"] for c in llm.async_calls)


@pytest.mark.asyncio
async def test_solve_async_calls_generate_async(fake_llm_factory):
    expected_sub_answer = "async_sub_answer_test"
    llm = fake_llm_factory({"sub_answer": expected_sub_answer})

    ltm = LeastToMost(llm, use_json_parsing=False)
    solved = await ltm.solve_async("main q", ["sub1", "sub2"])

    assert solved == [("sub1", expected_sub_answer), ("sub2", expected_sub_answer)]
    assert len(llm.async_calls) == 2
    assert all(c["type"] == "generate_async" for c in llm.async_calls)
