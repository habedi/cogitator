from typing import List, Optional, Union

from pydantic import BaseModel, Field


class LTMDecomposition(BaseModel):
    subquestions: List[str] = Field(..., description="List of sequential subquestions")


class ThoughtExpansion(BaseModel):
    thoughts: List[str] = Field(..., description="List of distinct reasoning steps or thoughts")


class EvaluationResult(BaseModel):
    score: int = Field(..., description="Quality score from 1 to 10", ge=1, le=10)
    justification: str = Field(..., description="Brief justification for the score")


class ExtractedAnswer(BaseModel):
    final_answer: Optional[Union[str, int, float]] = Field(
        ..., description="The final extracted answer"
    )
