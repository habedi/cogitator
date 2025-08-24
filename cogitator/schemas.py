"""Defines Pydantic models for structured data exchange within Cogitator."""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class LTMDecomposition(BaseModel):
    """Schema for the output of the Least-to-Most decomposition step."""

    subquestions: List[str] = Field(..., description="List of sequential subquestions")


class ThoughtExpansion(BaseModel):
    """Schema for the output of a thought expansion step (e.g., in ToT)."""

    thoughts: List[str] = Field(..., description="List of distinct reasoning steps or thoughts")


class EvaluationResult(BaseModel):
    """Schema for the output of an evaluation step (e.g., in ToT, GoT)."""

    score: int = Field(..., description="Quality score from 1 to 10")
    justification: str = Field(..., description="Brief justification for the score")


class ExtractedAnswer(BaseModel):
    """Schema for the final extracted answer from a reasoning chain."""

    final_answer: Optional[Union[str, int, float]] = Field(
        ..., description="The final extracted answer"
    )


class TraceNode(BaseModel):
    """Represents a single node in a reasoning trace for visualization or debugging."""

    node_id: Union[str, int] = Field(..., description="Unique identifier for the node.")
    parent_id: Optional[Union[str, int]] = Field(None, description="Identifier of the parent node.")
    content: str = Field(..., description="The textual content of the thought or step.")
    score: Optional[float] = Field(None, description="The evaluation score of the node.")
    visits: Optional[int] = Field(None, description="The number of visits (e.g., in MCTS).")
    # Generic metadata field for strategy-specific information
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata for the node."
    )


class Trace(BaseModel):
    """Represents the full execution trace of a reasoning strategy."""

    root_node_id: Union[str, int] = Field(..., description="The ID of the root node of the trace.")
    nodes: List[TraceNode] = Field(
        default_factory=list, description="A list of all nodes in the trace."
    )

    def add_node(
        self,
        node_id: Union[str, int],
        parent_id: Optional[Union[str, int]],
        content: str,
        score: Optional[float] = None,
        visits: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Adds a new node to the trace."""
        self.nodes.append(
            TraceNode(
                node_id=node_id,
                parent_id=parent_id,
                content=content,
                score=score,
                visits=visits,
                metadata=metadata,
            )
        )
