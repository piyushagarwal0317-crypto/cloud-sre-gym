"""
CloudScaleRL FastAPI Server
=============================
Exposes the RL environment over HTTP so the LLM agent (client.py)
can interact with it via POST /reset, POST /step, GET /state, GET /render.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from cloudscalerl.models import CloudScaleAction, CloudScaleObservation, CloudScaleReward
from cloudscalerl.server.cloudscalerl_env import CloudScaleEnvServer

# Load task configs from openenv.yaml
_OPENENV_PATH = Path(__file__).parent.parent / "openenv.yaml"
with open(_OPENENV_PATH) as f:
    _OPENENV = yaml.safe_load(f)

TASK_CONFIGS: dict[str, dict[str, Any]] = {
    t["id"]: t for t in _OPENENV["tasks"]
}

app = FastAPI(
    title="CloudScaleRL",
    description="Kubernetes-inspired cloud autoscaling RL environment",
    version="0.1.0",
)

# One active environment instance per server process (single-agent use)
_env: Optional[CloudScaleEnvServer] = None


# ── Request / Response Models ─────────────────────────────────────────────────


class ResetRequest(BaseModel):
    task_id: str
    seed: int = 42


class StepResponse(BaseModel):
    observation: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any]


# ── Routes ────────────────────────────────────────────────────────────────────


@app.post("/reset", response_model=dict[str, Any])
def reset(req: ResetRequest) -> dict[str, Any]:
    """Reset the environment for a new episode."""
    global _env

    if req.task_id not in TASK_CONFIGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{req.task_id}'. Valid: {list(TASK_CONFIGS.keys())}",
        )

    config = TASK_CONFIGS[req.task_id]
    _env = CloudScaleEnvServer()
    obs = _env._reset(task_id=req.task_id, seed=req.seed)
    return obs.model_dump()

@app.post("/step", response_model=StepResponse)
def step(action: CloudScaleAction) -> StepResponse:
    """Advance the environment one tick with the given action."""
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialised. Call /reset first.")

    obs, reward, done, info = _env._step(action)

    return StepResponse(
        observation=obs.model_dump(),
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/state", response_model=dict[str, Any])
def get_state() -> dict[str, Any]:
    """Get current environment state without advancing."""
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialised. Call /reset first.")
    obs = _env._build_observation()
    return obs.model_dump()


@app.get("/render", response_model=dict[str, Any])
def render() -> dict[str, Any]:
    """JSON dashboard snapshot — human-readable cluster state for debugging."""
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialised. Call /reset first.")
    return _env.render()


@app.get("/tasks", response_model=list[dict[str, Any]])
def list_tasks() -> list[dict[str, Any]]:
    """List all available tasks with metadata."""
    return list(TASK_CONFIGS.values())


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": "0.1.0"}