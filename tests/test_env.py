"""
Tests for CodeReviewEnv — covers graders, env API, and all 3 tasks.
Run with:  python -m pytest tests/ -v
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from models import CodeReviewAction
from env import CodeReviewEnv
from graders.grader import grade
from tasks.task_definitions import ALL_TASKS, TASK_EASY, TASK_MEDIUM, TASK_HARD


# ── Grader unit tests ─────────────────────────────────────────────────────────

class TestGraderEasy:
    def test_perfect_answer(self):
        action = CodeReviewAction(
            issues_found=["off-by-one error in range(): using len(numbers)+1 causes IndexError on line 4"],
            severity_tags=["high"],
            explanation="Fix by changing range(len(numbers)+1) to range(len(numbers)). Line 4 has a +1 that causes index out of range.",
            line_numbers=[4],
        )
        reward, info = grade(action, TASK_EASY)
        assert reward >= 0.75, f"Expected high reward for correct answer, got {reward}"
        assert info["issue_score"] == 1.0

    def test_wrong_answer(self):
        action = CodeReviewAction(
            issues_found=["missing docstring"],
            severity_tags=["low"],
            explanation="The function needs better documentation.",
        )
        reward, info = grade(action, TASK_EASY)
        assert reward < 0.4, f"Expected low reward for wrong answer, got {reward}"
        assert info["issue_score"] == 0.0

    def test_partial_answer(self):
        action = CodeReviewAction(
            issues_found=["index error when accessing the list"],
            severity_tags=["medium"],
            explanation="There might be an indexing problem. Should fix the loop.",
        )
        reward, info = grade(action, TASK_EASY)
        assert 0.2 <= reward <= 0.8, f"Expected partial reward, got {reward}"

    def test_reward_in_range(self):
        action = CodeReviewAction(issues_found=[], severity_tags=[], explanation="")
        reward, _ = grade(action, TASK_EASY)
        assert 0.0 <= reward <= 1.0


class TestGraderMedium:
    def test_finds_sql_injection(self):
        action = CodeReviewAction(
            issues_found=["SQL injection vulnerability via f-string on line 10"],
            severity_tags=["critical"],
            explanation="Use parameterized queries instead of string formatting. The f-string allows injection.",
        )
        reward, info = grade(action, TASK_MEDIUM)
        assert info["issue_score"] > 0.2

    def test_finds_all_three(self):
        action = CodeReviewAction(
            issues_found=[
                "SQL injection on line 10 via f-string query",
                "Hardcoded password and secret key on lines 3 and 4",
                "Plaintext password comparison on line 16 — use bcrypt hashing",
            ],
            severity_tags=["critical", "critical", "high"],
            explanation=(
                "1. Use parameterized queries to prevent SQL injection. "
                "2. Move credentials to env vars. "
                "3. Hash passwords with bcrypt before storing/comparing."
            ),
            line_numbers=[10, 3, 16],
        )
        reward, info = grade(action, TASK_MEDIUM)
        assert info["issue_score"] == 1.0
        assert reward >= 0.75


class TestGraderHard:
    def test_finds_eval(self):
        action = CodeReviewAction(
            issues_found=["eval() on line 18 allows arbitrary code execution"],
            severity_tags=["critical"],
            explanation="Replace eval() with ast.literal_eval() for safe parsing.",
        )
        reward, info = grade(action, TASK_HARD)
        assert info["issues_hit"][0] is True

    def test_all_issues(self):
        action = CodeReviewAction(
            issues_found=[
                "eval() on line 18 allows arbitrary code execution",
                "Insecure default fallback API key on line 4",
                "File handle resource leak in load_config line 7 — not using context manager",
                "Reading entire file into memory on line 17 — no size limit",
                "No error handling or input validation in save_results",
                "PEP 8 violation: multiple imports on line 1",
            ],
            severity_tags=["critical", "high", "medium", "medium", "medium", "low"],
            explanation=(
                "Critical: replace eval with ast.literal_eval. "
                "High: remove fallback default from env.get(). "
                "Medium: use 'with open()' context manager. "
                "Medium: add file size guard before reading. "
                "Medium: wrap save in try/except. "
                "Low: split imports onto separate lines per PEP 8."
            ),
            line_numbers=[18, 4, 7, 17, 24, 1],
        )
        reward, info = grade(action, TASK_HARD)
        assert info["issue_score"] == 1.0
        assert reward >= 0.80


# ── Environment API tests ─────────────────────────────────────────────────────

class TestEnvAPI:
    def test_invalid_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            CodeReviewEnv(task_name="nonexistent-task")

    def test_all_tasks_exist(self):
        for name in ["spot-the-bug", "security-audit", "full-review"]:
            env = CodeReviewEnv(task_name=name)
            assert env is not None

    def test_reset_returns_observation(self):
        async def _run():
            env = CodeReviewEnv(task_name="spot-the-bug")
            result = await env.reset()
            assert result.observation.code_snippet != ""
            assert result.observation.step_number == 0
            assert result.done is False
        asyncio.run(_run())

    def test_step_after_reset(self):
        async def _run():
            env = CodeReviewEnv(task_name="spot-the-bug")
            await env.reset()
            action = CodeReviewAction(
                issues_found=["off-by-one error in range on line 4"],
                severity_tags=["high"],
                explanation="Change range(len(numbers)+1) to range(len(numbers)).",
            )
            result = await env.step(action)
            assert 0.0 <= result.reward <= 1.0
            assert isinstance(result.done, bool)
        asyncio.run(_run())

    def test_state_returns_dict(self):
        async def _run():
            env = CodeReviewEnv(task_name="security-audit")
            await env.reset()
            state = await env.state()
            assert state["task_name"] == "security-audit"
            assert "step_count" in state
            assert "done" in state
        asyncio.run(_run())

    def test_episode_ends_at_max_steps(self):
        async def _run():
            env = CodeReviewEnv(task_name="spot-the-bug")
            await env.reset()
            action = CodeReviewAction(
                issues_found=["nothing"],
                severity_tags=["low"],
                explanation="x",
            )
            done = False
            steps = 0
            while not done:
                result = await env.step(action)
                done = result.done
                steps += 1
                if steps > 20:
                    break
            assert steps <= TASK_EASY.max_steps + 1
        asyncio.run(_run())

    def test_step_after_done_returns_error(self):
        async def _run():
            env = CodeReviewEnv(task_name="spot-the-bug")
            await env.reset()
            await env.close()
            action = CodeReviewAction(issues_found=[], severity_tags=[], explanation="")
            result = await env.step(action)
            assert result.last_action_error is not None
        asyncio.run(_run())


# ── Reward range sanity ───────────────────────────────────────────────────────

class TestRewardRange:
    @pytest.mark.parametrize("task_name", ALL_TASKS.keys())
    def test_reward_always_in_range(self, task_name):
        spec = ALL_TASKS[task_name]
        for _ in range(10):
            action = CodeReviewAction(
                issues_found=["random issue"],
                severity_tags=["low"],
                explanation="some explanation with fix suggestion",
            )
            reward, _ = grade(action, spec)
            assert 0.0 <= reward <= 1.0, f"Reward {reward} out of range for {task_name}"
