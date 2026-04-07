"""
Typed Pydantic models for the Code Review OpenEnv environment.
Defines Action, Observation, and StepResult per OpenEnv spec.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CodeReviewAction(BaseModel):
    """
    Action submitted by the agent each step.

    Fields:
        issues_found:   List of issues the agent identified (e.g. ["null pointer on line 5"])
        severity_tags:  Per-issue severity label: "low" | "medium" | "high" | "critical"
        explanation:    Free-text rationale / suggested fix for the overall snippet
        line_numbers:   Optional line numbers the agent associates with each issue
    """
    issues_found: List[str] = Field(default_factory=list, description="Issues identified in the code snippet")
    severity_tags: List[str] = Field(default_factory=list, description="Severity for each issue: low/medium/high/critical")
    explanation: str = Field(default="", description="Overall explanation and suggested fixes")
    line_numbers: Optional[List[int]] = Field(default=None, description="Line numbers associated with each issue")


class CodeReviewObservation(BaseModel):
    """
    Observation returned to the agent after reset() or step().

    Fields:
        code_snippet:       The code the agent must review
        language:           Programming language of the snippet
        task_description:   Human-readable task instructions
        step_number:        Current step within the episode
        last_feedback:      Feedback from the previous step (empty on reset)
        cumulative_score:   Running score so far in [0, 1]
        hints:              Optional hints (provided on harder tasks)
    """
    code_snippet: str = Field(..., description="Code snippet to review")
    language: str = Field(..., description="Programming language")
    task_description: str = Field(..., description="What the agent must do")
    step_number: int = Field(default=0)
    last_feedback: str = Field(default="", description="Feedback from previous step")
    cumulative_score: float = Field(default=0.0)
    hints: List[str] = Field(default_factory=list)


class StepResult(BaseModel):
    """Return value of env.step()."""
    observation: CodeReviewObservation
    reward: float = Field(..., ge=0.0, le=1.0)
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
    last_action_error: Optional[str] = None


class ResetResult(BaseModel):
    """Return value of env.reset()."""
    observation: CodeReviewObservation
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)
