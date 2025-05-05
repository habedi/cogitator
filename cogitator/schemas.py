"""Defines Pydantic models for structured data exchange within Cogitator."""

from typing import List, Optional, Union

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
