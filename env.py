"""
CodeReviewEnv — OpenEnv-compliant environment for AI code review tasks.

API surface (OpenEnv spec):
  env = await CodeReviewEnv.create(task_name, docker_image=None)
  result: ResetResult  = await env.reset()
  result: StepResult   = await env.step(action: CodeReviewAction)
  state:  dict         = await env.state()
  await env.close()
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional

from models import (
    CodeReviewAction,
    CodeReviewObservation,
    ResetResult,
    StepResult,
)
from tasks.task_definitions import ALL_TASKS, TaskSpec
from graders.grader import grade


class CodeReviewEnv:

    ENV_NAME = "code-review-env"
    VERSION = "1.0.0"

    def __init__(self, task_name: str = "spot-the-bug") -> None:
        if task_name not in ALL_TASKS:
            raise ValueError(
                f"Unknown task '{task_name}'. "
                f"Available tasks: {list(ALL_TASKS.keys())}"
            )
        self._task_name = task_name
        self._spec: TaskSpec = ALL_TASKS[task_name]

        # Episode state
        self._step_count: int = 0
        self._cumulative_reward: float = 0.0
        self._done: bool = False
        self._last_feedback: str = ""
        self._history: list[dict] = []
        self._start_time: float = time.time()

    # ── Factory methods ──

    @classmethod
    async def create(
        cls,
        task_name: str = "spot-the-bug",
        docker_image: Optional[str] = None,  
    ) -> "CodeReviewEnv":
        """Async factory — mirrors OpenEnv from_docker_image pattern."""
        instance = cls(task_name=task_name)
        return instance

    @classmethod
    async def from_docker_image(
        cls,
        image_name: Optional[str] = None,
        task_name: str = "spot-the-bug",
    ) -> "CodeReviewEnv":
        """Alias used in the sample inference script pattern."""
        return await cls.create(task_name=task_name, docker_image=image_name)

    # ── Core OpenEnv API ──

    async def reset(self) -> ResetResult:
        """Reset episode state and return initial observation."""
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._done = False
        self._last_feedback = ""
        self._history = []
        self._start_time = time.time()

        obs = self._make_observation()
        return ResetResult(observation=obs, done=False, info={"task": self._task_name})

    async def step(self, action: CodeReviewAction) -> StepResult:
        """
        Process one agent action.

        Returns reward proportional to how well the agent identified issues,
        labelled severities, and explained fixes.  Episode ends when:
          - max_steps reached, OR
          - agent achieves perfect score (reward == 1.0)
        """
        if self._done:
            obs = self._make_observation()
            return StepResult(
                observation=obs,
                reward=0.0,
                done=True,
                info={"warning": "Episode already done. Call reset()."},
                last_action_error="Episode already finished",
            )

        self._step_count += 1

        # Grade the action
        reward, grade_info = grade(action, self._spec)
        self._cumulative_reward = min(self._cumulative_reward + reward, 1.0)

        # Build feedback for the next observation
        self._last_feedback = self._build_feedback(action, grade_info)
        self._history.append(
            {
                "step": self._step_count,
                "action": action.model_dump(),
                "reward": reward,
                "grade_info": grade_info,
            }
        )

        # Episode termination
        done = (
            self._step_count >= self._spec.max_steps
            or reward >= 1.0
        )
        self._done = done

        obs = self._make_observation()
        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info={
                **grade_info,
                "step": self._step_count,
                "cumulative_reward": self._cumulative_reward,
            },
        )

    async def state(self) -> Dict[str, Any]:
        """Return current episode state (OpenEnv spec requirement)."""
        return {
            "env_name": self.ENV_NAME,
            "version": self.VERSION,
            "task_name": self._task_name,
            "difficulty": self._spec.difficulty,
            "step_count": self._step_count,
            "max_steps": self._spec.max_steps,
            "cumulative_reward": self._cumulative_reward,
            "done": self._done,
            "elapsed_seconds": round(time.time() - self._start_time, 2),
        }

    async def close(self) -> None:
        """Clean up resources (no-op for in-process env, required by spec)."""
        self._done = True

    # ── Sync helpers (convenience) ───
    def reset_sync(self) -> ResetResult:
        return asyncio.get_event_loop().run_until_complete(self.reset())

    def step_sync(self, action: CodeReviewAction) -> StepResult:
        return asyncio.get_event_loop().run_until_complete(self.step(action))

    # ── Internal helpers ───

    def _make_observation(self) -> CodeReviewObservation:
        hints = self._spec.hints if self._spec.difficulty in ("medium", "hard") else []
        return CodeReviewObservation(
            code_snippet=self._spec.code_snippet,
            language=self._spec.language,
            task_description=self._spec.description,
            step_number=self._step_count,
            last_feedback=self._last_feedback,
            cumulative_score=round(self._cumulative_reward, 4),
            hints=hints,
        )

    def _build_feedback(self, action: CodeReviewAction, grade_info: dict) -> str:
        issue_score = grade_info["issue_score"]
        n_expected = len(self._spec.ground_truth_issues)
        n_found = int(round(issue_score * n_expected))

        parts = [
            f"Step {self._step_count} feedback:",
            f"  Issues identified: {n_found}/{n_expected} "
            f"(score {grade_info['issue_score']:.2f})",
            f"  Severity accuracy: {grade_info['severity_score']:.2f}",
            f"  Explanation depth: {grade_info['explanation_score']:.2f}",
            f"  Step reward:       {grade_info['total_reward']:.2f}",
        ]

        if issue_score < 1.0:
            parts.append(
                "  Hint: re-read the snippet carefully — there may be more issues."
            )
        if grade_info["severity_score"] < 0.5:
            parts.append(
                "  Hint: reconsider your severity ratings (low/medium/high/critical)."
            )

        return "\n".join(parts)
