"""
DataBridgeEnv — FastAPI server.

Endpoints (all required by OpenEnv spec + hackathon checklist):
  POST /reset          → ContractObservation
  POST /step           → {observation, reward}
  GET  /state          → BridgeState
  GET  /tasks          → task list + action schema
  POST /grader         → last episode's grader score
  POST /baseline       → run baseline inference on all tasks
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from baseline import run_episode
from environment import DataBridgeEnvironment
from models import BridgeState, ContractObservation, ContractReward, TransformAction


app = FastAPI(
    title="DataBridgeEnv",
    description=(
        "An RL environment for debugging data contract violations "
        "between connected services. Part of the OpenEnv hackathon."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One shared environment instance (single-agent, stateful)
env = DataBridgeEnvironment()


# ---------------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: Optional[int] = None  # None = random seed each call


class StepResponse(BaseModel):
    observation: ContractObservation
    reward: ContractReward
    done: bool
    info: Dict[str, Any]


class GraderRequest(BaseModel):
    episode_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Core OpenEnv endpoints
# ---------------------------------------------------------------------------

@app.post("/reset", response_model=ContractObservation)
def reset(req: ResetRequest = None):
    """Start a new episode. Returns initial observation."""
    task_id = req.task_id if req else "easy"
    seed = req.seed if req else None
    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    try:
        obs = env.reset(task_id=task_id, seed=seed)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return obs


@app.post("/step", response_model=StepResponse)
def step(action: TransformAction):
    """Submit a transformed payload. Returns new observation + reward."""
    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state", response_model=BridgeState)
def state():
    """Return current episode state / metadata."""
    return env.state()


# ---------------------------------------------------------------------------
# Required extra endpoints (hackathon pre-submission checklist)
# ---------------------------------------------------------------------------

@app.get("/tasks")
def tasks():
    """
    Return the list of all tasks and the action schema.
    Required by OpenEnv spec for automated validation.
    """
    return {
        "tasks": [
            {
                "task_id": "easy",
                "name": "Field name & type mismatch",
                "difficulty": "easy",
                "description": (
                    "Frontend sends an order payload with camelCase field names "
                    "and string-encoded numbers. Fix field names and cast types."
                ),
                "max_steps": 3,
                "mismatch_types": ["field_name", "type_coercion"],
            },
            {
                "task_id": "medium",
                "name": "Shape, array, and missing field mismatch",
                "difficulty": "medium",
                "description": (
                    "Payload has a nested user object (should be flat), items "
                    "as a comma-separated string (should be a list), a missing "
                    "required field, and an extra unexpected field."
                ),
                "max_steps": 4,
                "mismatch_types": ["shape_nesting", "array_scalar", "missing_extra", "type_coercion"],
            },
            {
                "task_id": "hard",
                "name": "All 5 mismatch types — infer schema from examples",
                "difficulty": "hard",
                "description": (
                    "No explicit schema. The agent must infer the target contract "
                    "from three (broken, correct) example pairs, then fix the live "
                    "broken payload which contains all five mismatch types."
                ),
                "max_steps": 5,
                "mismatch_types": ["field_name", "type_coercion", "shape_nesting", "array_scalar", "missing_extra"],
            },
        ],
        "action_schema": {
            "type": "object",
            "description": "POST to /step with this body.",
            "properties": {
                "transformed_payload": {
                    "type": "object",
                    "description": "The agent's corrected payload (dict).",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Optional: chain-of-thought explaining the fix.",
                    "nullable": True,
                },
            },
            "required": ["transformed_payload"],
        },
    }


@app.post("/grader")
def grader(_req: GraderRequest = None):
    """
    Return the grader score for the most recently completed episode.
    Required by the hackathon pre-submission checklist.
    """
    s = env.state()
    return {
        "episode_id": s.episode_id,
        "task_id": s.task_id,
        "current_score": s.current_score,
        "step_number": s.step_number,
        "done": s.done,
    }


@app.post("/baseline")
def baseline_endpoint(request: Request):
    """
    Fast inline baseline endpoint.
    Runs exactly 1 episode per task with deterministic rule-based fallback.
    """
    task_ids = ["easy", "medium", "hard"]
    base_url = os.environ.get("ENV_BASE_URL", str(request.base_url).rstrip("/"))
    seed_start = 42

    results: Dict[str, Dict[str, Any]] = {}

    for idx, task_id in enumerate(task_ids):
        scores = []
        errors = []
        try:
            score = run_episode(
                task_id=task_id,
                openai_client=None,
                verbose=False,
                seed=seed_start + idx,
                base_url=base_url,
            )
            scores.append(score)
        except Exception as e:
            scores.append(0.0)
            errors.append(str(e))

        mean_score = round(sum(scores) / len(scores), 4) if scores else 0.0
        if len(errors) == len(scores):
            task_status = "error"
        elif errors:
            task_status = "partial"
        else:
            task_status = "ok"

        results[task_id] = {
            "scores": scores,
            "mean": mean_score,
            "status": task_status,
        }

    statuses = [results[t]["status"] for t in task_ids]
    if all(s == "ok" for s in statuses):
        overall_status = "ok"
    elif all(s == "error" for s in statuses):
        overall_status = "error"
    else:
        overall_status = "partial"

    return {
        "status": overall_status,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok"}


def main():
    reload = os.getenv("RELOAD", "false").lower() in {"1", "true", "yes"}
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=reload)


if __name__ == "__main__":
    main()
