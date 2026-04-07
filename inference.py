"""
Runs an LLM agent against all three code-review tasks and emits
structured stdout logs in the mandatory [START] / [STEP] / [END] format.

Environment variables (set before running):
  API_BASE_URL   LLM endpoint   (default: HuggingFace router)
  MODEL_NAME     Model ID       (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN       API key
  TASK_NAME      Override single task (default: runs all three)
"""

from __future__ import annotations

import asyncio
import json
import os
import textwrap
import re
import urllib.request
from typing import List, Optional

from openai import OpenAI

from env import CodeReviewEnv
from models import CodeReviewAction

# ── Config ──
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK    = "code-review-env"
TEMPERATURE  = 0.3   # lower = more deterministic for code review
MAX_TOKENS   = 400
SUCCESS_SCORE_THRESHOLD = 0.5

ALL_TASK_NAMES = ["spot-the-bug", "security-audit", "full-review"]

# ── Logging helpers ───

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action!r} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

def hf_request(url, headers, payload):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )

    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode())
    
    
# ── Prompts ──

SYSTEM_PROMPT = textwrap.dedent("""
   
    Respond ONLY with valid JSON matching this exact schema:
    {
      "issues_found": ["description of issue 1", "description of issue 2"],
      "severity_tags": ["high", "critical"],
      "line_numbers": [4, 10],
      "explanation": "Overall summary with fixes..."
    }

    No markdown, no code blocks, no extra text — pure JSON only.
""").strip()


def build_user_prompt(obs_dict: dict, step: int, last_feedback: str) -> str:
    parts = [
        f"Language: {obs_dict['language']}",
        f"Task: {obs_dict['task_description']}",
        "",
        "Code to review:",
        "```",
        obs_dict["code_snippet"],
        "```",
    ]
    if last_feedback:
        parts += ["", "Previous step feedback:", last_feedback]
    if obs_dict.get("hints"):
        parts += ["", "Hints: " + "; ".join(obs_dict["hints"])]
    parts += ["", f"Step {step}: Provide your full code review as JSON."]
    return "\n".join(parts)


# ── LLM call ───

def get_model_action(client, obs_dict, step, last_feedback):
    user_prompt = build_user_prompt(obs_dict, step, last_feedback)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        raw = completion.choices[0].message.content

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise Exception("No JSON found")

        data = json.loads(match.group())

        return CodeReviewAction(
            issues_found=data.get("issues_found", []),
            severity_tags=data.get("severity_tags", []),
            explanation=data.get("explanation", ""),
            line_numbers=data.get("line_numbers", None),
        )

    except Exception as exc:
        print(f"[DEBUG] Error: {exc}")

        return CodeReviewAction(
            issues_found=["Fallback issue"],
            severity_tags=["low"],
            explanation="Fallback response",
        )

# ── Episode runner ──

async def run_episode(client, task_name: str) -> float:
    """Run one full episode for a task. Returns normalized score in [0,1]."""
    env = await CodeReviewEnv.create(task_name=task_name)
    from tasks.task_definitions import ALL_TASKS
    spec = ALL_TASKS[task_name]

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_result = await env.reset()
        obs = reset_result.observation
        last_feedback = ""

        for step in range(1, spec.max_steps + 1):
            action = get_model_action(client, obs.model_dump(), step, last_feedback)

            # Compact action string for the log line
            action_str = (
                f"issues={len(action.issues_found)} "
                f"severity={action.severity_tags} "
                f"explanation_len={len(action.explanation)}"
            )

            step_result = await env.step(action)
            reward   = step_result.reward
            done     = step_result.done
            error    = step_result.last_action_error

            rewards.append(reward)
            steps_taken = step
            obs = step_result.observation
            last_feedback = obs.last_feedback

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        score = max(rewards) if rewards else 0.0
        score = round(min(max(score, 0.0), 1.0), 3)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ── Main ──
async def main() -> None:
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY
    )

    task_override = os.getenv("TASK_NAME")
    tasks_to_run  = [task_override] if task_override else ALL_TASK_NAMES

    scores = {}
    for task_name in tasks_to_run:
        print(f"\n{'='*60}", flush=True)
        score = await run_episode(client, task_name)
        scores[task_name] = score

    print("\n" + "="*60, flush=True)
    print("[SUMMARY]", flush=True)
    for task, s in scores.items():
        print(f"  {task}: {s:.3f}", flush=True)
    overall = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"  overall_avg: {overall:.3f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())