"""Tests for cogitator.schemas module."""

from cogitator.schemas import (
    EvaluationResult,
    ExtractedAnswer,
    LTMDecomposition,
    ThoughtExpansion,
    Trace,
    TraceNode,
)


def test_ltm_decomposition_schema():
    """Test LTMDecomposition validation and fields."""
    schema = LTMDecomposition(subquestions=["Q1", "Q2"])
    assert schema.subquestions == ["Q1", "Q2"]


def test_thought_expansion_schema():
    """Test ThoughtExpansion validation and fields."""
    schema = ThoughtExpansion(thoughts=["thought 1", "thought 2"])
    assert schema.thoughts == ["thought 1", "thought 2"]


def test_evaluation_result_schema():
    """Test EvaluationResult validation and fields."""
    schema = EvaluationResult(score=8, justification="Clear reasoning.")
    assert schema.score == 8
    assert schema.justification == "Clear reasoning."


def test_extracted_answer_schema():
    """Test ExtractedAnswer validation and fields."""
    schema = ExtractedAnswer(final_answer="42")
    assert schema.final_answer == "42"

    schema_null = ExtractedAnswer(final_answer=None)
    assert schema_null.final_answer is None


def test_trace_and_trace_node():
    """Test Trace and TraceNode serialization and methods."""
    node = TraceNode(
        node_id=1,
        parent_id=None,
        content="root thought",
        score=9.5,
        visits=5,
        metadata={"key": "val"}
    )
    assert node.node_id == 1
    assert node.parent_id is None
    assert node.content == "root thought"
    assert node.score == 9.5
    assert node.visits == 5
    assert node.metadata == {"key": "val"}

    trace = Trace(root_node_id=1)
    assert trace.root_node_id == 1
    assert len(trace.nodes) == 0

    trace.add_node(
        node_id=2,
        parent_id=1,
        content="child thought",
        score=8.0,
        visits=2,
        metadata={"step": 1}
    )
    assert len(trace.nodes) == 1
    added_node = trace.nodes[0]
    assert added_node.node_id == 2
    assert added_node.parent_id == 1
    assert added_node.content == "child thought"
    assert added_node.score == 8.0
    assert added_node.visits == 2
    assert added_node.metadata == {"step": 1}
