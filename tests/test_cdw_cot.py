import pytest

from cogitator.cdw_cot import CDWCoT


def test_init_pool_builds_candidate_pool(fake_llm_factory, patch_embedding_clustering):
    questions = [f"q{i}" for i in range(8)]
    answers = [f"a{i}" for i in range(8)]
    llm = fake_llm_factory({"generate_sync": "Pool CoT Sync"})
    cdw = CDWCoT(llm, pool_size=4, n_clusters=2, sample_size=2)
    cdw.init_pool(questions, answers)
    assert len(cdw.PC) <= 4
    assert cdw.cluster_centers is not None
    assert len(cdw.p_cluster) == cdw.cluster_centers.shape[0]
    if cdw.PC:
        assert "Pool CoT Sync" in cdw.PC[0]
        for p in cdw.p_cluster:
            assert pytest.approx(1.0) == p.sum()


@pytest.mark.asyncio
async def test_init_pool_async_builds_candidate_pool(fake_llm_factory, patch_embedding_clustering):
    questions = [f"q{i}" for i in range(8)]
    answers = [f"a{i}" for i in range(8)]
    llm = fake_llm_factory({"generate_async": "Pool CoT Async"})
    cdw = CDWCoT(llm, pool_size=4, n_clusters=2, sample_size=2)
    await cdw.init_pool_async(questions, answers)
    assert len(cdw.PC) <= 4
    assert cdw.cluster_centers is not None
    assert len(cdw.p_cluster) == cdw.cluster_centers.shape[0]
    if cdw.PC:
        assert "Pool CoT Async" in cdw.PC[0]
        for p in cdw.p_cluster:
            assert pytest.approx(1.0) == p.sum()


def test_train_and_answer_flow_runs_without_error(fake_llm_factory, patch_embedding_clustering):
    questions = [f"q{i}{i}" for i in range(10)]  # More data
    answers = [f"a{i}" for i in range(10)]
    llm = fake_llm_factory({"generate_sync": "Train/Answer Sync"})
    cdw = CDWCoT(llm, pool_size=5, n_clusters=3, sample_size=2, lr=0.1, temp=1.0)
    cdw.init_pool(questions, answers)
    cdw.train(val_split=0.3, epochs=1, patience=1)
    if not cdw.PC or cdw.cluster_centers is None: pytest.skip("Pool/Clusters empty")
    out = cdw.answer("some sync test")
    assert out == "Train/Answer Sync"
    assert "some sync test" in llm.sync_calls[-1]["prompt"]


@pytest.mark.asyncio
async def test_train_async_and_answer_async_flow(fake_llm_factory, patch_embedding_clustering):
    questions = [f"q{i}{i}" for i in range(10)]
    answers = [f"a{i}" for i in range(10)]
    llm = fake_llm_factory({"generate_async": "Train/Answer Async"})
    cdw = CDWCoT(llm, pool_size=5, n_clusters=3, sample_size=2, lr=0.1, temp=1.0, seed=42)
    await cdw.init_pool_async(questions, answers)
    await cdw.train_async(val_split=0.3, epochs=1, patience=1)
    if not cdw.PC or cdw.cluster_centers is None: pytest.skip("Pool/Clusters empty")
    out = await cdw.answer_async("some async test")
    assert out == "Train/Answer Async"
    assert "some async test" in llm.async_calls[-1]["prompt"]


def test_init_pool_raises_on_length_mismatch(fake_llm_factory):
    llm = fake_llm_factory()
    cdw = CDWCoT(llm)
    with pytest.raises(ValueError):
        cdw.init_pool(["q1"], ["a1", "a2"])


@pytest.mark.asyncio
async def test_init_pool_async_raises_on_length_mismatch(fake_llm_factory):
    llm = fake_llm_factory()
    cdw = CDWCoT(llm)
    with pytest.raises(ValueError):
        await cdw.init_pool_async(["q1"], ["a1", "a2"])
