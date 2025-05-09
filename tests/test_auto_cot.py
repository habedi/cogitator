import pytest

from cogitator import AutoCoT


def test_fit_builds_expected_number_of_demos(fake_llm_factory, patch_embedding_clustering):
    questions = [f"q{i}" for i in range(8)]
    llm = fake_llm_factory({
        "generate_sync": "Step 1\nStep 2"
    })
    ac = AutoCoT(llm, n_demos=2, max_q_tokens=100, max_steps=5)
    ac.fit(questions)
    assert ac.demos is not None
    assert len(ac.demos) == 2
    for demo in ac.demos:
        assert demo.startswith("Q: ")
        assert "Step 1" in demo


@pytest.mark.asyncio
async def test_fit_async_builds_expected_number_of_demos(fake_llm_factory,
                                                         patch_embedding_clustering):
    questions = [f"q{i}" for i in range(8)]
    llm = fake_llm_factory({
        "generate_async": "Async Step 1\nAsync Step 2"
    })
    ac = AutoCoT(llm, n_demos=2, max_q_tokens=100, max_steps=5)
    await ac.fit_async(questions)
    assert ac.demos is not None
    assert len(ac.demos) == 2
    for demo in ac.demos:
        assert demo.startswith("Q: ")
        assert "Async Step 1" in demo


def test_run_uses_cached_demos_and_constructs_payload(fake_llm_factory,
                                                      patch_embedding_clustering):
    questions = [f"q{i}" for i in range(8)]
    llm = fake_llm_factory({
        "generate_sync": "Sync Final Answer"
    })
    ac = AutoCoT(llm, n_demos=2)
    ac.fit(questions)
    assert ac.demos is not None
    out = ac.run("test question")
    assert out == "Sync Final Answer"
    assert "test question" in llm.sync_calls[-1]["prompt"]


@pytest.mark.asyncio
async def test_run_async_uses_cached_demos(fake_llm_factory, patch_embedding_clustering):
    questions = [f"q{i}" for i in range(8)]
    llm = fake_llm_factory({
        "generate_async": "Async Final Answer"
    })
    ac = AutoCoT(llm, n_demos=2)
    await ac.fit_async(questions)
    assert ac.demos is not None
    out = await ac.run_async("test question async")
    assert out == "Async Final Answer"
    assert "test question async" in llm.async_calls[-1]["prompt"]


def test_fit_raises_with_insufficient_questions(fake_llm_factory):
    llm = fake_llm_factory()
    ac = AutoCoT(llm, n_demos=3)
    with pytest.raises(ValueError):
        ac.fit(["only one"])


@pytest.mark.asyncio
async def test_fit_async_raises_with_insufficient_questions(fake_llm_factory):
    llm = fake_llm_factory()
    ac = AutoCoT(llm, n_demos=3)
    with pytest.raises(ValueError):
        await ac.fit_async(["only one"])
