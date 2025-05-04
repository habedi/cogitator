import asyncio

import pytest

from cogitator import ExtractedAnswer
from cogitator import SelfConsistency


def test_extract_answer_heuristic_with_equal_sign():
    sc = SelfConsistency(llm=None)
    assert sc._extract_answer_heuristic("…\n4+1 = 5") == "5"
    assert sc._extract_answer_heuristic("Result=42.") == "42"


def test_extract_answer_heuristic_with_prefixes():
    sc = SelfConsistency(llm=None)
    assert sc._extract_answer_heuristic("Therefore, the answer is 7.") == "7"
    assert sc._extract_answer_heuristic("Answer: 99") == "99"
    assert sc._extract_answer_heuristic("Ans 13") == "13"
    assert sc._extract_answer_heuristic("### 123") == "123"
    assert sc._extract_answer_heuristic("Final Answer: foo bar") == "foo bar"


def test_extract_answer_heuristic_last_line_fallback():
    sc = SelfConsistency(llm=None)
    cot = "Step1\nSome thought\nFinal numerical answer 100"
    assert sc._extract_answer_heuristic(cot) == "Final numerical answer 100"


def test_extract_answer_heuristic_numeric_last_line():
    sc = SelfConsistency(llm=None)
    cot = "Step1\nSome thought\n12345"
    assert sc._extract_answer_heuristic(cot) == "12345"
    cot_dollar = "Step1\nSome thought\n$56.78"
    assert sc._extract_answer_heuristic(cot_dollar) == "56.78"


def test_run_majority_vote_heuristic(fake_llm_factory):
    responses = ["...\nAns 10", "...\nFinal Answer: 10", "...\nResult=5"]
    llm = fake_llm_factory({"generate_sync": responses})
    sc = SelfConsistency(llm=llm, n_samples=3)  # Defaults to heuristic
    out = sc.run("dummy prompt")
    assert out == "10"
    assert len(llm.sync_calls) == 3


def test_extract_answer_json_path(fake_llm_factory):
    expected_answer = "AnswerFromJSON"
    llm = fake_llm_factory({
        "json_answer": ExtractedAnswer(final_answer=expected_answer)
    })
    sc = SelfConsistency(llm=llm, internal_extraction_format="json")
    extracted = sc.extract_answer("Some reasoning text...")
    assert extracted == expected_answer
    assert len(llm.sync_calls) == 1
    assert llm.sync_calls[0]["type"] == "_generate_json_internal"
    assert llm.sync_calls[0]["response_model"] == "ExtractedAnswer"


def test_run_majority_vote_json(fake_llm_factory):
    cot_responses = ["CoT leading to 10", "Another CoT leading to 10", "CoT leading to 5"]
    extraction_responses = [
        ExtractedAnswer(final_answer="10"),
        ExtractedAnswer(final_answer="10"),
        ExtractedAnswer(final_answer="5")
    ]
    # Configure mock: generate CoTs, then respond to extraction prompts
    llm = fake_llm_factory({
        "generate_sync": cot_responses,
        "json_answer": extraction_responses  # Mock should pick this for JSON extraction calls
    })
    sc = SelfConsistency(llm=llm, n_samples=3, internal_extraction_format="json")
    out = sc.run("dummy prompt for json run")
    assert out == "10"  # Verify majority vote result
    assert len(llm.sync_calls) == 6  # 3 generate + 3 json_internal
    assert sum(1 for c in llm.sync_calls if c["type"] == "generate") == 3
    assert sum(1 for c in llm.sync_calls if c["type"] == "_generate_json_internal") == 3


def test_run_stream_not_implemented(fake_llm_factory):
    llm = fake_llm_factory()
    sc = SelfConsistency(llm=llm)
    with pytest.raises(NotImplementedError):
        next(sc.run_stream("anything"))


@pytest.mark.asyncio
async def test_extract_answer_async_heuristic():
    sc = SelfConsistency(llm=None)
    assert await sc.extract_answer_async("...\nFinal Answer: ABC_async") == "ABC_async"
    assert await sc.extract_answer_async("...\nResult=55_async") == "55_async"


@pytest.mark.asyncio
async def test_run_async_majority_vote_heuristic(fake_llm_factory):
    responses = ["...\nAns Async10", "...\nFinal Answer: Async10", "...\nResult=Async5"]
    llm = fake_llm_factory({"generate_async": responses})
    sc = SelfConsistency(llm=llm, n_samples=3)
    out = await sc.run_async("dummy async prompt")
    assert out == "Async10"
    assert len(llm.async_calls) == 3


@pytest.mark.asyncio
async def test_extract_answer_json_async_path(fake_llm_factory):
    expected_answer = "AsyncAnswerFromJSON"
    llm = fake_llm_factory({
        "json_answer": ExtractedAnswer(final_answer=expected_answer)
    })
    sc = SelfConsistency(llm=llm, internal_extraction_format="json")
    extracted = await sc.extract_answer_async("Some async reasoning text...")
    assert extracted == expected_answer
    assert len(llm.async_calls) == 1
    assert llm.async_calls[0]["type"] == "_generate_json_internal_async"
    assert llm.async_calls[0]["response_model"] == "ExtractedAnswer"


@pytest.mark.asyncio
async def test_run_async_majority_vote_json(fake_llm_factory):
    cot_responses_async = ["Async CoT 10", "Async CoT 10 again", "Async CoT 5"]
    extraction_responses_async = [
        ExtractedAnswer(final_answer="10"),
        ExtractedAnswer(final_answer="10"),
        ExtractedAnswer(final_answer="5")
    ]
    llm = fake_llm_factory({
        "generate_async": cot_responses_async,
        "json_answer": extraction_responses_async
    })
    sc = SelfConsistency(llm=llm, n_samples=3, internal_extraction_format="json")
    out = await sc.run_async("dummy async json run prompt")
    assert out == "10"  # Verify majority vote result
    assert len(llm.async_calls) == 6  # 3 generate_async + 3 json_internal_async
    assert sum(1 for c in llm.async_calls if c["type"] == "generate_async") == 3
    assert sum(1 for c in llm.async_calls if c["type"] == "_generate_json_internal_async") == 3


@pytest.mark.asyncio
async def test_run_async_with_semaphore(fake_llm_factory):
    responses = ["…\nAnswer: S1", "…\nAnswer: S2", "…\nAnswer: S1"]
    llm = fake_llm_factory({"generate_async": responses})
    sc = SelfConsistency(llm=llm, n_samples=3)
    semaphore = asyncio.Semaphore(2)
    out = await sc.run_async("dummy async prompt semaphore", semaphore=semaphore)
    assert out == "S1"
    assert len(llm.async_calls) == 3


@pytest.mark.asyncio
async def test_run_stream_async_not_implemented(fake_llm_factory):
    llm = fake_llm_factory()
    sc = SelfConsistency(llm=llm)
    with pytest.raises(NotImplementedError):
        # Corrected: await the coroutine
        await sc.run_stream_async("anything")
