"""
Pre-Submission Validator
========================
Checks ALL competition requirements before you submit.
Run this locally (with dependencies installed) before pushing to HF Spaces.

Usage:
    python validate.py

All checks must pass (✓) or you risk disqualification.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

PASS = "  ✓"
FAIL = "  ✗"
WARN = "  ⚠"

errors: list[str] = []
warnings: list[str] = []


def check(label: str, condition: bool, error_msg: str = "", warn: bool = False) -> bool:
    if condition:
        print(f"{PASS} {label}")
        return True
    else:
        sym = WARN if warn else FAIL
        print(f"{sym} {label}")
        if error_msg:
            print(f"      → {error_msg}")
        if warn:
            warnings.append(label)
        else:
            errors.append(label)
        return False


# ── 1. File structure ─────────────────────────────────────────────────────────
print("\n[1/6] File structure")
required_files = [
    "models.py",
    "env.py",
    "server.py",
    "inference.py",
    "openenv.yaml",
    "Dockerfile",
    "requirements.txt",
    "README.md",
    "tasks/task_definitions.py",
    "graders/grader.py",
    "tests/test_env.py",
]
for f in required_files:
    check(f, (ROOT / f).exists(), f"Missing file: {f}")


# ── 2. openenv.yaml ───────────────────────────────────────────────────────────
print("\n[2/6] openenv.yaml")
try:
    import yaml
    with open(ROOT / "openenv.yaml") as fh:
        cfg = yaml.safe_load(fh)
    check("Has 'name' field", "name" in cfg)
    check("Has 'tasks' field", "tasks" in cfg)
    check("Has >= 3 tasks", len(cfg.get("tasks", [])) >= 3,
          f"Only {len(cfg.get('tasks', []))} tasks defined")
    check("Has 'action_space' field", "action_space" in cfg)
    check("Has 'observation_space' field", "observation_space" in cfg)
    check("Has 'reward_function' field", "reward_function" in cfg)
except ImportError:
    check("yaml importable", False, "pip install pyyaml")
except Exception as e:
    check("openenv.yaml parseable", False, str(e))


# ── 3. OpenEnv spec compliance ────────────────────────────────────────────────
print("\n[3/6] OpenEnv spec compliance")
try:
    from models import CodeReviewAction, CodeReviewObservation, StepResult, ResetResult
    check("CodeReviewAction importable", True)
    check("CodeReviewObservation importable", True)
    check("StepResult importable", True)
    check("ResetResult importable", True)

    from env import CodeReviewEnv

    async def _test_api():
        env = await CodeReviewEnv.create("spot-the-bug")
        r = await env.reset()
        assert hasattr(r, "observation"), "reset() must return ResetResult with .observation"
        assert hasattr(r, "done"), "reset() must return ResetResult with .done"

        action = CodeReviewAction(
            issues_found=["off-by-one error in range on line 4"],
            severity_tags=["high"],
            explanation="Fix the range boundary to avoid IndexError.",
        )
        sr = await env.step(action)
        assert hasattr(sr, "reward"), "step() must return StepResult with .reward"
        assert hasattr(sr, "done"), "step() must return StepResult with .done"
        assert hasattr(sr, "observation"), "step() must return StepResult with .observation"
        assert 0.0 <= sr.reward <= 1.0, f"reward {sr.reward} out of [0,1]"

        state = await env.state()
        assert isinstance(state, dict), "state() must return dict"
        assert "task_name" in state, "state() must include task_name"

        await env.close()
        return sr.reward

    reward = asyncio.run(_test_api())
    check("env.reset() works", True)
    check("env.step() works", True)
    check("env.state() works", True)
    check("env.close() works", True)
    check("reward in [0, 1]", True)

except Exception as e:
    check("OpenEnv API", False, str(e))


# ── 4. Tasks and graders ──────────────────────────────────────────────────────
print("\n[4/6] Tasks and graders")
try:
    from tasks.task_definitions import ALL_TASKS
    from graders.grader import grade

    check("At least 3 tasks defined", len(ALL_TASKS) >= 3,
          f"Only {len(ALL_TASKS)} tasks")

    difficulties = [t.difficulty for t in ALL_TASKS.values()]
    check("Has 'easy' task", "easy" in difficulties)
    check("Has 'medium' task", "medium" in difficulties)
    check("Has 'hard' task", "hard" in difficulties)

    # Check grader returns varying scores (not constant)
    from models import CodeReviewAction
    scores = set()
    for spec in ALL_TASKS.values():
        bad = CodeReviewAction(issues_found=[], severity_tags=[], explanation="")
        good = CodeReviewAction(
            issues_found=[spec.ground_truth_issues[0][:40]],
            severity_tags=[spec.severity_answers[0]],
            explanation="Fix: " + spec.ground_truth_issues[0][:60],
        )
        r_bad, _ = grade(bad, spec)
        r_good, _ = grade(good, spec)
        scores.add(round(r_bad, 2))
        scores.add(round(r_good, 2))

    check("Grader produces varying scores (not constant)", len(scores) > 2,
          "Grader always returns same score — disqualifying")

    # All rewards in range
    all_in_range = all(0.0 <= s <= 1.0 for s in scores)
    check("All grader scores in [0.0, 1.0]", all_in_range, f"Scores: {scores}")

except Exception as e:
    check("Tasks and graders", False, str(e))


# ── 5. Dockerfile ─────────────────────────────────────────────────────────────
print("\n[5/6] Dockerfile")
try:
    dockerfile = (ROOT / "Dockerfile").read_text()
    check("EXPOSE 7860", "EXPOSE 7860" in dockerfile)
    check("CMD or ENTRYPOINT present", "CMD" in dockerfile or "ENTRYPOINT" in dockerfile)
    check("Uses python base image", "python" in dockerfile.lower())
    check("Copies requirements.txt", "requirements.txt" in dockerfile)
    check("Installs requirements", "pip install" in dockerfile)
except Exception as e:
    check("Dockerfile readable", False, str(e))


# ── 6. inference.py format ────────────────────────────────────────────────────
print("\n[6/6] inference.py")
try:
    inf_text = (ROOT / "inference.py").read_text()
    check("[START] format present", "[START]" in inf_text)
    check("[STEP] format present", "[STEP]" in inf_text)
    check("[END] format present", "[END]" in inf_text)
    check("Uses OpenAI client", "OpenAI" in inf_text)
    check("Reads HF_TOKEN env var", "HF_TOKEN" in inf_text)
    check("Reads API_BASE_URL env var", "API_BASE_URL" in inf_text)
    check("Reads MODEL_NAME env var", "MODEL_NAME" in inf_text)
    check("reward formatted .2f", ":.2f" in inf_text)
    check("score formatted .3f or .2f", ":.3f" in inf_text or "score" in inf_text)
except Exception as e:
    check("inference.py readable", False, str(e))


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
if not errors:
    print(f"✅  All checks passed! ({len(warnings)} warnings)")
    print("    You're ready to push to Hugging Face Spaces.")
else:
    print(f"❌  {len(errors)} check(s) FAILED — fix before submitting:")
    for e in errors:
        print(f"    • {e}")
if warnings:
    print(f"\n⚠   {len(warnings)} warning(s):")
    for w in warnings:
        print(f"    • {w}")
print()
sys.exit(0 if not errors else 1)
