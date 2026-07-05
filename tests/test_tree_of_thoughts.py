import asyncio
from typing import Any
import pytest
from cogitator import EvaluationResult, ThoughtExpansion, TreeOfThoughts


def test_run_returns_final_and_calls_prompts(fake_llm_factory: Any) -> None:
    fake_expansion = ThoughtExpansion(thoughts=["step1_sync"])
    fake_eval = EvaluationResult(score=8, justification="Okay_sync")
    llm = fake_llm_factory(
        {"json_steps": fake_expansion, "json_eval": fake_eval, "final_answer": "FINAL_sync"}
    )
    tot = TreeOfThoughts(llm, max_depth=1, num_branches=1, sims=1, c_puct=1.0)
    out = tot.run("test?")

    assert out == "FINAL_sync"

    expand_call = next(
        (
            c
            for c in llm.sync_calls
            if c["type"] == "_generate_json_internal"
            and "JSON Output:" in c["prompt"]
            and "thoughts" in c["prompt"]
        ),
        None,
    )
    eval_call = next(
        (
            c
            for c in llm.sync_calls
            if c["type"] == "_generate_json_internal" and "JSON Evaluation:" in c["prompt"]
        ),
        None,
    )
    final_call = next(
        (
            c
            for c in llm.sync_calls
            if c["type"] == "generate"
            and (
                "Given reasoning steps" in c["prompt"]
                or c["prompt"].startswith("Answer the question:")
            )
        ),
        None,
    )

    assert expand_call is not None, "Expansion call not found"
    assert expand_call["response_model"] == "ThoughtExpansion"
    assert eval_call is not None, "Evaluation call not found"
    assert eval_call["response_model"] == "EvaluationResult"
    assert final_call is not None, "Final answer generation call not found"


@pytest.mark.asyncio
async def test_run_async_returns_final_and_calls_prompts(fake_llm_factory: Any) -> None:
    fake_expansion_async = ThoughtExpansion(thoughts=["step1_async"])
    fake_eval_async = EvaluationResult(score=8, justification="Okay_async")
    llm = fake_llm_factory(
        {
            "json_steps": fake_expansion_async,
            "json_eval": fake_eval_async,
            "final_answer": "FINAL_async",
        }
    )
    tot = TreeOfThoughts(llm, max_depth=1, num_branches=1, sims=1, c_puct=1.0)
    out = await tot.run_async("test_async?")

    assert out == "FINAL_async"

    expand_call = next(
        (
            c
            for c in llm.async_calls
            if c["type"] == "_generate_json_internal_async"
            and "JSON Output:" in c["prompt"]
            and "thoughts" in c["prompt"]
        ),
        None,
    )
    eval_call = next(
        (
            c
            for c in llm.async_calls
            if c["type"] == "_generate_json_internal_async" and "JSON Evaluation:" in c["prompt"]
        ),
        None,
    )
    final_call = next(
        (
            c
            for c in llm.async_calls
            if c["type"] == "generate_async"
            and (
                "Given reasoning steps" in c["prompt"]
                or c["prompt"].startswith("Answer the question:")
            )
        ),
        None,
    )

    assert expand_call is not None, "Async expansion call not found"
    assert expand_call["response_model"] == "ThoughtExpansion"
    assert eval_call is not None, "Async evaluation call not found"
    assert eval_call["response_model"] == "EvaluationResult"
    assert final_call is not None, "Async final answer generation call not found"


def test_node_representation() -> None:
    from cogitator.strategies.tree_of_thoughts import TreeOfThoughts

    node = TreeOfThoughts._Node(steps=["step1", "step2"])
    assert "Node(" in repr(node)
    assert "steps=2" in repr(node)


def test_select_unvisited_children() -> None:
    from cogitator.strategies.tree_of_thoughts import TreeOfThoughts

    tot = TreeOfThoughts(None)  # type: ignore
    root = TreeOfThoughts._Node([])
    child = TreeOfThoughts._Node(["step1"], parent=root)
    root.children.append(child)
    selected = tot._select(root)
    assert selected is child


def test_expand_failures(fake_llm_factory: Any) -> None:
    # Case 1: generate_json returns something else
    llm = fake_llm_factory()
    llm.generate_json = lambda *args, **kwargs: "not a ThoughtExpansion"
    tot = TreeOfThoughts(llm, seed=42)
    node = TreeOfThoughts._Node([])
    tot._expand(node, "question")
    assert len(node.children) == 0

    # Case 2: generate_json raises exception
    llm2 = fake_llm_factory()
    llm2.generate_json = lambda *args, **kwargs: (_ for _ in ()).throw(Exception("fail"))
    tot2 = TreeOfThoughts(llm2, seed=42)
    node2 = TreeOfThoughts._Node([])
    tot2._expand(node2, "question")
    assert len(node2.children) == 0


@pytest.mark.asyncio
async def test_expand_async_failures(fake_llm_factory: Any) -> None:
    llm = fake_llm_factory()

    async def mock_return_str(*args: Any, **kwargs: Any) -> str:
        return "not a ThoughtExpansion"

    llm.generate_json_async = mock_return_str
    tot = TreeOfThoughts(llm)
    node = TreeOfThoughts._Node([])
    await tot._expand_async(node, "question", semaphore=None)
    assert len(node.children) == 0

    async def mock_raise(*args: Any, **kwargs: Any) -> None:
        raise Exception("fail async")

    llm.generate_json_async = mock_raise
    await tot._expand_async(node, "question", semaphore=None)
    assert len(node.children) == 0


def test_evaluate_failures(fake_llm_factory: Any) -> None:
    llm = fake_llm_factory()
    llm.generate_json = lambda *args, **kwargs: "not an EvaluationResult"
    tot = TreeOfThoughts(llm)
    node = TreeOfThoughts._Node(["step"])
    score = tot._evaluate(node, "question")
    assert score == 0.0

    llm2 = fake_llm_factory()
    llm2.generate_json = lambda *args, **kwargs: (_ for _ in ()).throw(Exception("fail"))
    tot2 = TreeOfThoughts(llm2)
    score2 = tot2._evaluate(node, "question")
    assert score2 == 0.0


@pytest.mark.asyncio
async def test_evaluate_async_failure(fake_llm_factory: Any) -> None:
    llm = fake_llm_factory()

    async def mock_fail(*args: Any, **kwargs: Any) -> None:
        raise Exception("fail")

    llm.generate_json_async = mock_fail
    tot = TreeOfThoughts(llm)
    node = TreeOfThoughts._Node(["step"])
    score = await tot._evaluate_async(node, "question", semaphore=None)
    assert score == 0.0


def test_trace_updates_and_backprop(fake_llm_factory: Any) -> None:
    fake_expansion = ThoughtExpansion(thoughts=["step1"])
    fake_eval = EvaluationResult(score=10, justification="Perfect")
    llm = fake_llm_factory(
        {"json_steps": fake_expansion, "json_eval": fake_eval, "final_answer": "ans"}
    )
    tot = TreeOfThoughts(llm, max_depth=1, num_branches=1, sims=1)
    ans, trace = tot.run("question", with_trace=True)
    assert ans == "ans"
    assert trace is not None
    assert len(trace.nodes) > 0


def test_run_no_thoughts_generated(fake_llm_factory: Any) -> None:
    llm = fake_llm_factory({"final_answer": "direct answer"})
    llm.generate_json = lambda *args, **kwargs: ThoughtExpansion(thoughts=[])
    tot = TreeOfThoughts(llm, sims=1)
    ans = tot.run("question")
    assert ans == "direct answer"


@pytest.mark.asyncio
async def test_run_async_no_thoughts_generated(fake_llm_factory: Any) -> None:
    llm = fake_llm_factory({"final_answer": "direct answer async"})

    async def mock_empty_expand(*args: Any, **kwargs: Any) -> ThoughtExpansion:
        return ThoughtExpansion(thoughts=[])

    llm.generate_json_async = mock_empty_expand
    tot = TreeOfThoughts(llm, sims=1)
    ans = await tot.run_async("question")
    assert ans == "direct answer async"


def test_run_final_answer_generation_fails(fake_llm_factory: Any) -> None:
    fake_expansion = ThoughtExpansion(thoughts=["step1"])
    fake_eval = EvaluationResult(score=8, justification="OK")
    llm = fake_llm_factory({"json_steps": fake_expansion, "json_eval": fake_eval})
    llm.generate = lambda *args, **kwargs: (_ for _ in ()).throw(Exception("final LLM fail"))
    tot = TreeOfThoughts(llm, max_depth=1, num_branches=1, sims=1)
    ans = tot.run("question")
    assert ans == "Error generating final answer."


@pytest.mark.asyncio
async def test_run_async_final_answer_generation_fails(fake_llm_factory: Any) -> None:
    fake_expansion = ThoughtExpansion(thoughts=["step1"])
    fake_eval = EvaluationResult(score=8, justification="OK")
    llm = fake_llm_factory({"json_steps": fake_expansion, "json_eval": fake_eval})

    async def mock_fail(*args: Any, **kwargs: Any) -> None:
        raise Exception("final async LLM fail")

    llm.generate_async = mock_fail
    tot = TreeOfThoughts(llm, max_depth=1, num_branches=1, sims=1)
    ans = await tot.run_async("question")
    assert ans == "Error generating final async answer."


def test_run_stream_not_implemented(fake_llm_factory: Any) -> None:
    llm = fake_llm_factory()
    tot = TreeOfThoughts(llm)
    with pytest.raises(NotImplementedError):
        tot.run_stream("prompt")


@pytest.mark.asyncio
async def test_run_stream_async_not_implemented(fake_llm_factory: Any) -> None:
    llm = fake_llm_factory()
    tot = TreeOfThoughts(llm)
    with pytest.raises(NotImplementedError):
        await tot.run_stream_async("prompt")


def test_init_with_seed(fake_llm_factory: Any) -> None:
    llm = fake_llm_factory()
    tot = TreeOfThoughts(llm, seed=42)
    assert tot.seed == 42


def test_evaluate_metadata_none(fake_llm_factory: Any) -> None:
    from cogitator.schemas import Trace

    llm = fake_llm_factory({"json_eval": EvaluationResult(score=8, justification="Good")})
    tot = TreeOfThoughts(llm)
    node = TreeOfThoughts._Node(["step1"])
    trace = Trace(root_node_id=0)
    trace.add_node(node_id=node.id, parent_id=None, content="step1", metadata=None)
    tot._evaluate(node, "question", trace=trace)
    assert trace.nodes[0].metadata == {"justification": "Good"}


@pytest.mark.asyncio
async def test_evaluate_async_metadata_none(fake_llm_factory: Any) -> None:
    from cogitator.schemas import Trace

    expected_eval = EvaluationResult(score=8, justification="Good Async")
    llm = fake_llm_factory({"json_eval": expected_eval})
    tot = TreeOfThoughts(llm)
    node = TreeOfThoughts._Node(["step1"])
    trace = Trace(root_node_id=0)
    trace.add_node(node_id=node.id, parent_id=None, content="step1", metadata=None)
    await tot._evaluate_async(node, "question", semaphore=None, trace=trace)
    assert trace.nodes[0].metadata == {"justification": "Good Async"}


def test_select_traversal(fake_llm_factory: Any) -> None:
    fake_expansion = ThoughtExpansion(thoughts=["step1", "step2"])
    fake_eval = EvaluationResult(score=8, justification="Ok")
    llm = fake_llm_factory(
        {"json_steps": fake_expansion, "json_eval": fake_eval, "final_answer": "ans"}
    )
    tot = TreeOfThoughts(llm, max_depth=2, num_branches=2, sims=2)
    ans = tot.run("question")
    assert ans == "ans"


@pytest.mark.asyncio
async def test_expand_and_evaluate_async_with_semaphore(fake_llm_factory: Any) -> None:
    from cogitator.schemas import Trace

    fake_expansion = ThoughtExpansion(thoughts=["step1"])
    fake_eval = EvaluationResult(score=8, justification="Ok")
    llm = fake_llm_factory(
        {"json_steps": fake_expansion, "json_eval": fake_eval, "final_answer": "ans"}
    )
    tot = TreeOfThoughts(llm, max_depth=1, num_branches=1, sims=1)
    sem = asyncio.Semaphore(2)
    trace = Trace(root_node_id=0)

    node = TreeOfThoughts._Node([])
    await tot._expand_async(node, "question", semaphore=sem, trace=trace)
    assert len(node.children) == 1

    score = await tot._evaluate_async(node.children[0], "question", semaphore=sem, trace=trace)
    assert score > 0.0


def test_trace_loop_exits_normally_when_node_not_found(fake_llm_factory: Any) -> None:
    from cogitator.schemas import Trace

    llm = fake_llm_factory({"json_eval": EvaluationResult(score=8, justification="Good")})
    tot = TreeOfThoughts(llm)
    node = TreeOfThoughts._Node(["step1"])
    trace = Trace(root_node_id=0)
    tot._evaluate(node, "question", trace=trace)
    tot._backpropagate(node, 0.8, trace=trace)


@pytest.mark.asyncio
async def test_run_async_with_trace(fake_llm_factory: Any) -> None:
    fake_expansion = ThoughtExpansion(thoughts=["step1"])
    fake_eval = EvaluationResult(score=10, justification="Perfect")
    llm = fake_llm_factory(
        {"json_steps": fake_expansion, "json_eval": fake_eval, "final_answer": "ans"}
    )
    tot = TreeOfThoughts(llm, max_depth=1, num_branches=1, sims=1)
    ans, trace = await tot.run_async("question", with_trace=True)
    assert ans == "ans"
    assert trace is not None


def test_run_max_depth_zero(fake_llm_factory: Any) -> None:
    llm = fake_llm_factory({"final_answer": "ans"})
    tot = TreeOfThoughts(llm, max_depth=0, sims=1)
    ans = tot.run("question")
    assert ans == "ans"


@pytest.mark.asyncio
async def test_run_async_max_depth_zero(fake_llm_factory: Any) -> None:
    llm = fake_llm_factory({"final_answer": "ans"})
    tot = TreeOfThoughts(llm, max_depth=0, sims=1)
    ans = await tot.run_async("question")
    assert ans == "ans"


@pytest.mark.asyncio
async def test_run_async_with_semaphore(fake_llm_factory: Any) -> None:
    fake_expansion = ThoughtExpansion(thoughts=["step1"])
    fake_eval = EvaluationResult(score=10, justification="Perfect")
    llm = fake_llm_factory(
        {"json_steps": fake_expansion, "json_eval": fake_eval, "final_answer": "ans"}
    )
    tot = TreeOfThoughts(llm, max_depth=1, num_branches=1, sims=1)
    sem = asyncio.Semaphore(2)
    ans = await tot.run_async("question", semaphore=sem)
    assert ans == "ans"


@pytest.mark.asyncio
async def test_trace_loop_exits_normally_when_node_not_found_async(
    fake_llm_factory: Any,
) -> None:
    from cogitator.schemas import Trace

    llm = fake_llm_factory({"json_eval": EvaluationResult(score=8, justification="Good")})
    tot = TreeOfThoughts(llm)
    node = TreeOfThoughts._Node(["step1"])
    trace = Trace(root_node_id=0)
    await tot._evaluate_async(node, "question", semaphore=None, trace=trace)
