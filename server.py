"""
FastAPI server exposing CodeReviewEnv via HTTP.

Endpoints (OpenEnv spec):
  POST /reset         { "task_name": "spot-the-bug" }
  POST /step          { "issues_found": [...], "severity_tags": [...], ... }
  GET  /state
  POST /close
  GET  /health
  GET  /tasks         list available tasks

Run locally:
  uvicorn server:app --host 0.0.0.0 --port 7860 --reload
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env import CodeReviewEnv
from models import CodeReviewAction
from tasks.task_definitions import ALL_TASKS

app = FastAPI(
    title="Code Review OpenEnv",
    description="An OpenEnv environment where AI agents practice real-world code review.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared environment instance per server process
_env: Optional[CodeReviewEnv] = None


# ── Request / Response schemas

class ResetRequest(BaseModel):
    task_name: str = "spot-the-bug"


class StepRequest(BaseModel):
    issues_found: List[str] = []
    severity_tags: List[str] = []
    explanation: str = ""
    line_numbers: Optional[List[int]] = None


# ── Endpoints

@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "env": CodeReviewEnv.ENV_NAME, "version": CodeReviewEnv.VERSION}


@app.get("/tasks")
async def list_tasks() -> Dict[str, Any]:
    return {
        "tasks": [
            {
                "task_id": spec.task_id,
                "difficulty": spec.difficulty,
                "language": spec.language,
                "max_steps": spec.max_steps,
                "n_issues": len(spec.ground_truth_issues),
            }
            for spec in ALL_TASKS.values()
        ]
    }


@app.post("/reset")
async def reset(req: Optional[ResetRequest] = None) -> Dict[str, Any]:
    global _env

    # default task
    task_name = "spot-the-bug"

    if req and req.task_name:
        if req.task_name not in ALL_TASKS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown task '{req.task_name}'. Valid: {list(ALL_TASKS.keys())}",
            )
        task_name = req.task_name

    _env = await CodeReviewEnv.create(task_name=task_name)
    result = await _env.reset()

    return {
        "observation": result.observation.model_dump(),
        "done": result.done,
        "info": result.info,
    }


@app.post("/step")
async def step(req: StepRequest) -> Dict[str, Any]:
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")

    action = CodeReviewAction(
        issues_found=req.issues_found,
        severity_tags=req.severity_tags,
        explanation=req.explanation,
        line_numbers=req.line_numbers,
    )
    result = await _env.step(action)
    return {
        "observation": result.observation.model_dump(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info,
        "last_action_error": result.last_action_error,
    }


@app.get("/state")
async def state() -> Dict[str, Any]:
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    return await _env.state()


@app.post("/close")
async def close() -> Dict[str, str]:
    global _env
    if _env is not None:
        await _env.close()
        _env = None
    return {"status": "closed"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)
