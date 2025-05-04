import pytest

from cogitator import LTMDecomposition, ExtractedAnswer
from cogitator import LeastToMost


def test_decompose_numbered(fake_llm_factory):
    llm = fake_llm_factory({
        "json_subquestions": LTMDecomposition(subquestions=["sub1", "sub2", "sub3"])
    })
    ltm = LeastToMost(llm, intermediate_output_format="json")
    subs = ltm.decompose("anything")
    assert subs == ["sub1", "sub2", "sub3"]
    assert len(llm.sync_calls) == 1
    assert llm.sync_calls[0]["type"] == "_generate_json_internal"
    assert llm.sync_calls[0]["response_model"] == "LTMDecomposition"


def test_decompose_bullets(fake_llm_factory):
    llm = fake_llm_factory({
        "json_subquestions": LTMDecomposition(subquestions=["subA", "subB"])
    })
    ltm = LeastToMost(llm, intermediate_output_format="json")
    subs = ltm.decompose("anything")
    assert subs == ["subA", "subB"]


def test_decompose_fallback_simulated(fake_llm_factory):
    llm = fake_llm_factory({
        "json_subquestions": LTMDecomposition(
            subquestions=["This is a sentence", "Another sentence"])
    })
    ltm = LeastToMost(llm, intermediate_output_format="json")
    subs = ltm.decompose("anything")
    assert subs == ["This is a sentence", "Another sentence"]


def test_decompose_empty_list_raises(fake_llm_factory):
    llm = fake_llm_factory({
        "json_subquestions": LTMDecomposition(subquestions=[])
    })
    ltm = LeastToMost(llm, intermediate_output_format="json")
    with pytest.raises(ValueError, match="LLM returned empty subquestions list after validation."):
        ltm.decompose("anything")


def test_decompose_invalid_json_raises(fake_llm_factory):
    llm = fake_llm_factory()
    llm.generate_json = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("generate_json failed... Last error: JSONDecodeError..."))
    ltm = LeastToMost(llm, intermediate_output_format="json")
    with pytest.raises(ValueError,
                       match=r"Failed to decompose question due to LLM error: RuntimeError"):
        ltm.decompose("anything")


def test_decompose_invalid_schema_raises(fake_llm_factory):
    llm = fake_llm_factory()
    llm.generate_json = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("generate_json failed... Last error: ValidationError..."))
    ltm = LeastToMost(llm, intermediate_output_format="json")
    with pytest.raises(ValueError,
                       match=r"Failed to decompose question due to LLM error: RuntimeError"):
        ltm.decompose("anything")


def test_max_subqs_trims(fake_llm_factory):
    many_subs = [f"sub{i + 1}" for i in range(20)]
    llm = fake_llm_factory({
        "json_subquestions": LTMDecomposition(subquestions=many_subs)
    })
    ltm = LeastToMost(llm, max_subqs=5, intermediate_output_format="json")
    subs = ltm.decompose("anything")
    assert len(subs) == 5
    assert subs == ["sub1", "sub2", "sub3", "sub4", "sub5"]


@pytest.mark.asyncio
async def test_decompose_async_numbered(fake_llm_factory):
    llm = fake_llm_factory({
        "json_subquestions": LTMDecomposition(subquestions=["sub1", "sub2", "sub3"])
    })
    ltm = LeastToMost(llm, intermediate_output_format="json")
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
    ltm = LeastToMost(llm, intermediate_output_format="json")
    with pytest.raises(ValueError,
                       match="Async LLM returned empty subquestions list after validation."):
        await ltm.decompose_async("anything async")


@pytest.mark.asyncio
async def test_decompose_async_invalid_json_raises(fake_llm_factory):
    llm = fake_llm_factory()

    async def mock_fail(*args, **kwargs):
        raise RuntimeError("generate_json_async failed... Last error: JSONDecodeError...")

    llm.generate_json_async = mock_fail
    ltm = LeastToMost(llm, intermediate_output_format="json")
    with pytest.raises(ValueError,
                       match=r"Async decomposition failed due to LLM error: RuntimeError"):
        await ltm.decompose_async("anything async")


@pytest.mark.asyncio
async def test_decompose_async_invalid_schema_raises(fake_llm_factory):
    llm = fake_llm_factory()

    async def mock_fail(*args, **kwargs):
        raise RuntimeError("generate_json_async failed... Last error: ValidationError...")

    llm.generate_json_async = mock_fail
    ltm = LeastToMost(llm, intermediate_output_format="json")
    with pytest.raises(ValueError,
                       match=r"Async decomposition failed due to LLM error: RuntimeError"):
        await ltm.decompose_async("anything async")


# --- Run tests ---

def test_run_integration_text_mode(fake_llm_factory):
    llm = fake_llm_factory({
        "json_subquestions": LTMDecomposition(subquestions=["dummy sub1", "dummy sub2"]),
        "sub_answer": "sub_answer_sync_text",
        "final_answer": "ANS_sync_text"
    })
    ltm = LeastToMost(llm, intermediate_output_format="text")
    out = ltm.run("question?")
    assert out == "ANS_sync_text"
    assert any(
        c["type"] == "_generate_json_internal" and c["response_model"] == "LTMDecomposition" for c
        in llm.sync_calls)
    solve_calls = [c for c in llm.sync_calls if
                   c["type"] == "generate" and "Current Subquestion:" in c["prompt"]]
    final_call = next((c for c in llm.sync_calls if
                       c["type"] == "generate" and c["prompt"].startswith(
                           "Based on the following")), None)
    assert len(solve_calls) == 2
    assert final_call is not None


def test_run_integration_json_mode(fake_llm_factory):
    subquestions = ["dummy sub1", "dummy sub2"]
    question = "question?"
    fake_sub_answer_obj = ExtractedAnswer(final_answer="sub_answer_sync_json")
    fake_final_answer_obj = ExtractedAnswer(final_answer="ANS_sync_json")

    solve_prompt_key_1 = "Current Subquestion: dummy sub1"
    solve_prompt_key_2 = "Current Subquestion: dummy sub2"
    # The final prompt starts with "Based on..." and contains the original question
    final_prompt_key = f"Original Main Question: {question}"

    llm = fake_llm_factory({
        "json_subquestions": LTMDecomposition(subquestions=subquestions),
        "responses_map": {

            solve_prompt_key_1: fake_sub_answer_obj,
            solve_prompt_key_2: fake_sub_answer_obj,
            final_prompt_key: fake_final_answer_obj
        }
    })
    ltm = LeastToMost(llm, intermediate_output_format="json")
    out = ltm.run(question)
    assert out == "ANS_sync_json"

    assert any(
        c["type"] == "_generate_json_internal" and c["response_model"] == "LTMDecomposition" for c
        in llm.sync_calls)
    solve_json_calls = [c for c in llm.sync_calls if
                        c["type"] == "_generate_json_internal" and "Current Subquestion:" in c[
                            "prompt"]]
    final_json_call = next((c for c in llm.sync_calls if
                            c["type"] == "_generate_json_internal" and c["prompt"].startswith(
                                "Based on the following")), None)
    assert len(solve_json_calls) == 2
    assert final_json_call is not None


@pytest.mark.asyncio
async def test_run_async_integration_text_mode(fake_llm_factory):
    llm = fake_llm_factory({
        "json_subquestions": LTMDecomposition(
            subquestions=["dummy sub1 async", "dummy sub2 async"]),
        "sub_answer": "sub_answer_async_text",
        "final_answer": "ANS_async_text"
    })
    ltm = LeastToMost(llm, intermediate_output_format="text")
    out = await ltm.run_async("question async?")
    assert out == "ANS_async_text"
    assert any(
        c["type"] == "_generate_json_internal_async" and c["response_model"] == "LTMDecomposition"
        for c in llm.async_calls)
    solve_calls = [c for c in llm.async_calls if
                   c["type"] == "generate_async" and "Current Subquestion:" in c["prompt"]]
    final_call = next((c for c in llm.async_calls if
                       c["type"] == "generate_async" and c["prompt"].startswith(
                           "Based on the following")), None)
    assert len(solve_calls) == 2
    assert final_call is not None


@pytest.mark.asyncio
async def test_run_async_integration_json_mode(fake_llm_factory):
    subquestions = ["dummy sub1 async", "dummy sub2 async"]
    question = "question async?"
    fake_sub_answer_obj_async = ExtractedAnswer(final_answer="sub_answer_async_json")
    fake_final_answer_obj_async = ExtractedAnswer(final_answer="ANS_async_json")

    solve_prompt_key_1 = "Current Subquestion: dummy sub1 async"
    solve_prompt_key_2 = "Current Subquestion: dummy sub2 async"
    final_prompt_key = f"Original Main Question: {question}"  # Include the specific question

    llm = fake_llm_factory({
        "json_subquestions": LTMDecomposition(subquestions=subquestions),
        "responses_map": {
            solve_prompt_key_1: fake_sub_answer_obj_async,
            solve_prompt_key_2: fake_sub_answer_obj_async,
            final_prompt_key: fake_final_answer_obj_async
        }
    })
    ltm = LeastToMost(llm, intermediate_output_format="json")
    out = await ltm.run_async(question)
    assert out == "ANS_async_json"

    assert any(
        c["type"] == "_generate_json_internal_async" and c["response_model"] == "LTMDecomposition"
        for c in llm.async_calls)
    solve_json_calls = [c for c in llm.async_calls if
                        c["type"] == "_generate_json_internal_async" and "Current Subquestion:" in
                        c["prompt"]]
    final_json_call = next((c for c in llm.async_calls if
                            c["type"] == "_generate_json_internal_async" and c["prompt"].startswith(
                                "Based on the following")), None)
    assert len(solve_json_calls) == 2
    assert final_json_call is not None


# --- Solve tests ---
@pytest.mark.asyncio
async def test_solve_async_calls_generate_async_text(fake_llm_factory):
    expected_sub_answer = "async_sub_answer_test_text"
    llm = fake_llm_factory({"sub_answer": expected_sub_answer})
    ltm = LeastToMost(llm, intermediate_output_format="text")
    solved = await ltm.solve_async("main q", ["sub1", "sub2"])
    assert solved == [("sub1", expected_sub_answer), ("sub2", expected_sub_answer)]
    assert len(llm.async_calls) == 2
    assert all(c["type"] == "generate_async" for c in llm.async_calls)


@pytest.mark.asyncio
async def test_solve_async_calls_generate_json_async(fake_llm_factory):
    expected_sub_answer_obj = ExtractedAnswer(final_answer="async_sub_answer_test_json")
    llm = fake_llm_factory({"json_answer": expected_sub_answer_obj})
    ltm = LeastToMost(llm, intermediate_output_format="json")
    solved = await ltm.solve_async("main q", ["sub1", "sub2"])
    assert solved == [("sub1", "async_sub_answer_test_json"),
                      ("sub2", "async_sub_answer_test_json")]
    assert len(llm.async_calls) == 2
    assert all(c["type"] == "_generate_json_internal_async" for c in llm.async_calls)
    assert all(c["response_model"] == "ExtractedAnswer" for c in llm.async_calls)
